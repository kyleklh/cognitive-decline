import sys
from pathlib import Path

import joblib
import pandas as pd

# keep artifact paths relative to repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hmm_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
META_PATH = PROJECT_ROOT / "models" / "hmm_meta.pkl"

REQUIRED_COLS = ["d_t", "tau_t", "n_t", "v_t"]
DEFAULT_ROLE_MAP = {"normal": 0, "borderline": 1, "exit_seek": 2}

# inference must use the exact artifacts saved by train_hmm.py so scaling and state geometry stay consistent
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# keep fallback if meta file is missing
if META_PATH.exists():
    hmm_meta = joblib.load(META_PATH)
else:
    hmm_meta = {
        "state_role_map": DEFAULT_ROLE_MAP,
        "low_motion_threshold": None,
    }


def _validate_required_columns(df):
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def _estimate_dt(df):
    # use positive tau_t increments as a robust frame-to-time estimate when fps metadata is unavailable
    tau_step_candidates = df["tau_t"].diff()
    tau_step_candidates = tau_step_candidates[tau_step_candidates > 0]
    return float(tau_step_candidates.median()) if len(tau_step_candidates) > 0 else (1.0 / 6.0)


def _compute_role_ratios(df, role_map):
    # prefer soft posteriors from hmm instead of hard viterbi counts to reduce boundary jitter
    posterior = df.attrs.get("state_posterior")
    if posterior is None:
        counts = df["state"].value_counts()
        total = len(df)
        return {
            "normal": counts.get(role_map.get("normal", -1), 0) / total,
            "borderline": counts.get(role_map.get("borderline", -1), 0) / total,
            "exit_seek": counts.get(role_map.get("exit_seek", -1), 0) / total,
        }

    n_states = posterior.shape[1]
    ratios = {}
    for role_name in ("normal", "borderline", "exit_seek"):
        state_id = role_map.get(role_name, -1)
        if isinstance(state_id, int) and 0 <= state_id < n_states:
            ratios[role_name] = float(posterior[:, state_id].mean())
        else:
            ratios[role_name] = 0.0

    ratio_sum = sum(ratios.values())
    # normalize so downstream score blending stays comparable across clips
    if ratio_sum > 0:
        ratios = {k: v / ratio_sum for k, v in ratios.items()}
    return ratios


def _compute_role_scores(role_ratios, mean_abs_v, max_tau_t, near_exit_ratio, approaches_per_minute):
    # clip-level decision blends latent-state evidence with smooth exit cues from dwell and approach dynamics
    movement_scale = mean_abs_v / (mean_abs_v + 20.0)
    dwell_signal = max_tau_t / (max_tau_t + 5.0)
    entry_signal = approaches_per_minute / (approaches_per_minute + 1.5)

    normal_score = role_ratios["normal"] * (1.0 - near_exit_ratio) * (1.0 - 0.5 * movement_scale)
    borderline_score = (
        role_ratios["borderline"]
        * (1.0 + 0.2 * movement_scale)
        * (1.0 - 0.45 * entry_signal)
    )
    # keep exit score anchored to actual latent exit evidence; avoid overcalling from a single brief visit
    exit_score = (
        role_ratios["exit_seek"] * (1.0 + near_exit_ratio + dwell_signal + 0.8 * entry_signal)
        + 0.15 * entry_signal * role_ratios["borderline"]
        + 0.10 * near_exit_ratio * role_ratios["borderline"]
    )

    return {
        "normal": float(normal_score),
        "borderline": float(borderline_score),
        "exit_seek": float(exit_score),
    }


def _classify_behavior(role_scores, role_ratios, max_tau_t, near_exit_ratio, entry_count, meta):
    borderline_dwell_threshold = float(meta.get("borderline_dwell_threshold", 2.0))
    exit_dwell_threshold = float(meta.get("exit_dwell_threshold", 8.0))
    borderline_near_exit_threshold = float(meta.get("borderline_near_exit_ratio_threshold", 0.08))
    exit_near_exit_threshold = float(meta.get("exit_near_exit_ratio_threshold", 0.35))

    mild_exit_exposure = (entry_count > 0) or (near_exit_ratio > 0.0) or (max_tau_t > 0.0)
    weak_exit_hmm = role_ratios["exit_seek"] < 0.20

    in_borderline_band = (
        (max_tau_t >= borderline_dwell_threshold or near_exit_ratio >= borderline_near_exit_threshold)
        and max_tau_t < exit_dwell_threshold
        and near_exit_ratio < exit_near_exit_threshold
    )

    strong_exit_exposure = (
        max_tau_t >= exit_dwell_threshold
        or near_exit_ratio >= exit_near_exit_threshold
        or entry_count >= 3
    )

    # sustained exit interaction should remain exit-seeking even when latent state occupancy leans borderline
    if strong_exit_exposure and (
        role_ratios["normal"] < 0.85 or role_ratios["borderline"] > 0.25
    ):
        return "exit_seek"

    # treat sparse/brief exit interactions as borderline when latent exit evidence is weak
    if mild_exit_exposure and weak_exit_hmm:
        if in_borderline_band:
            return "borderline"
        if entry_count <= 2 and near_exit_ratio < exit_near_exit_threshold and max_tau_t < exit_dwell_threshold:
            return "borderline"

    return max(role_scores, key=role_scores.get)


def predict(csv_file):
    df = pd.read_csv(csv_file)
    _validate_required_columns(df)

    # keep missing-value handling same as training
    df = df.ffill().bfill().fillna(0)

    X = df[REQUIRED_COLS].values

    if len(X) == 0:
        raise ValueError("No valid rows found in input CSV after preprocessing.")

    # use the training scaler for inference
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    state_posterior = model.predict_proba(X_scaled)

    df["state"] = states
    df.attrs["state_posterior"] = state_posterior
    return df


def summarize_behavior(df):
    # print both views so discrepancies are easy to interpret when frame timeline and clip decision differ
    counts = df["state"].value_counts()
    total = len(df)

    role_map = hmm_meta.get("state_role_map", DEFAULT_ROLE_MAP)

    normal_state = role_map.get("normal", 0)
    borderline_state = role_map.get("borderline", 1)
    exit_seek_state = role_map.get("exit_seek", 2)

    role_ratios = _compute_role_ratios(df, role_map)
    normal = role_ratios["normal"]
    borderline = role_ratios["borderline"]
    exit_seek = role_ratios["exit_seek"]

    mean_abs_v = float(df["v_t"].abs().mean())
    max_tau_t = float(df["tau_t"].max())
    near_exit_ratio = float((df["tau_t"] > 0).mean())
    entry_count = int(df["n_t"].max())
    dt_est = _estimate_dt(df)
    clip_duration_seconds = max(total * dt_est, 1e-6)
    approaches_per_minute = float(entry_count * 60.0 / clip_duration_seconds)
    print("\n=== behavior summary ===")
    print("state distribution:")
    for state_id, count in counts.sort_index().items():
        share = (count / total) * 100.0
        print(f"  state {state_id}: {count} frames ({share:.1f}%)")

    print("\nclip stats:")
    print(f"  mean |v_t|           : {mean_abs_v:.3f}")
    print(f"  max tau_t            : {max_tau_t:.3f}")
    print(f"  near-exit ratio      : {near_exit_ratio:.3f}")
    print(f"  exit entries         : {entry_count}")
    print(f"  est clip duration(s) : {clip_duration_seconds:.2f}")
    print(f"  exit approaches/min  : {approaches_per_minute:.2f}")

    print("\nhmm role ratios:")
    print(f"  normal    : {normal:.3f}")
    print(f"  borderline: {borderline:.3f}")
    print(f"  exit_seek : {exit_seek:.3f}")

    role_scores = _compute_role_scores(
        role_ratios,
        mean_abs_v,
        max_tau_t,
        near_exit_ratio,
        approaches_per_minute,
    )
    print("\nrole scores (hmm + smooth evidence):")
    print(f"  normal    : {role_scores['normal']:.3f}")
    print(f"  borderline: {role_scores['borderline']:.3f}")
    print(f"  exit_seek : {role_scores['exit_seek']:.3f}")

    predicted_role = _classify_behavior(
        role_scores,
        role_ratios,
        max_tau_t,
        near_exit_ratio,
        entry_count,
        hmm_meta,
    )
    if predicted_role == "exit_seek":
        print("\nBehavior: EXIT-SEEKING (hmm evidence score)")
    elif predicted_role == "normal":
        print("\nBehavior: NORMAL movement (hmm evidence score)")
    else:
        print("\nBehavior: BORDERLINE exit-seeking (hmm evidence score)")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/hmm/detect_behavior.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    df = predict(csv_file)

    summarize_behavior(df)