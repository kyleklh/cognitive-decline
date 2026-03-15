import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


FEATURE_COLUMNS = ["d_t", "tau_t", "n_t", "v_t"]
N_COMPONENTS = 3
N_ITER = 200
BORDERLINE_Q = 0.75
EXIT_Q = 0.9


def _read_feature_csv(file_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    df = pd.read_csv(file_path)

    # fill occasional missing detections so short tracking gaps do not break sequence continuity
    df = df.ffill().bfill().fillna(0)

    sequence = df[FEATURE_COLUMNS].values
    stats = {
        "max_tau_t": float(df["tau_t"].max()),
        "near_exit_ratio": float((df["tau_t"] > 0).mean()),
    }
    return sequence, stats


def _calibrate_state_roles(
    model: hmm.GaussianHMM,
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
) -> Tuple[Dict[int, float], Dict[str, int], Dict[int, Dict[str, float]]]:
    # hidden-state ids are arbitrary so this maps fitted states to semantic roles used at inference time
    train_states = model.predict(x_scaled)
    abs_v = np.abs(x_raw[:, 3])
    tau_t = x_raw[:, 1]
    near_exit = tau_t > 0

    state_abs_v_mean: Dict[int, float] = {}
    state_stats: Dict[int, Dict[str, float]] = {}

    for state_id in range(model.n_components):
        mask = train_states == state_id
        if np.any(mask):
            mean_abs_v = float(np.mean(abs_v[mask]))
            mean_tau_t = float(np.mean(tau_t[mask]))
            near_exit_ratio = float(np.mean(near_exit[mask]))
        else:
            mean_abs_v = 0.0
            mean_tau_t = 0.0
            near_exit_ratio = 0.0

        state_abs_v_mean[state_id] = mean_abs_v
        state_stats[state_id] = {
            "mean_abs_v": mean_abs_v,
            "mean_tau_t": mean_tau_t,
            "near_exit_ratio": near_exit_ratio,
        }

    state_ids = list(range(model.n_components))

    # pick exit-seeking first using strongest exit interaction profile across tau_t, near_exit_ratio, and motion
    exit_state = max(
        state_ids,
        key=lambda s: (
            state_stats[s]["mean_tau_t"],
            state_stats[s]["near_exit_ratio"],
            state_stats[s]["mean_abs_v"],
        ),
    )

    # pick normal as the calmest remaining state then assign the leftover state as borderline
    remaining = [s for s in state_ids if s != exit_state]
    normal_state = min(
        remaining,
        key=lambda s: (
            state_stats[s]["mean_abs_v"],
            state_stats[s]["mean_tau_t"],
            state_stats[s]["near_exit_ratio"],
        ),
    )

    borderline_candidates = [s for s in remaining if s != normal_state]
    borderline_state = borderline_candidates[0] if borderline_candidates else normal_state

    state_role_map = {
        "normal": int(normal_state),
        "borderline": int(borderline_state),
        "exit_seek": int(exit_state),
    }
    return state_abs_v_mean, state_role_map, state_stats


def _compute_low_motion_threshold(abs_v: np.ndarray, lengths: Sequence[int]) -> float:
    # build a sequence-level baseline so near-static clips are not overcalled as exit-seeking
    seq_mean_abs_v: List[float] = []
    start = 0
    for length in lengths:
        end = start + length
        seq_mean_abs_v.append(float(np.mean(abs_v[start:end])))
        start = end
    return float(np.quantile(seq_mean_abs_v, 0.25))


def _quantile_thresholds(values: Sequence[float]) -> Tuple[float, float]:
    """Return (borderline, exit) thresholds from a value list"""
    return float(np.quantile(values, BORDERLINE_Q)), float(np.quantile(values, EXIT_Q))


def _build_meta(
    sequence_stats: Sequence[Dict[str, float]],
    state_abs_v_mean: Dict[int, float],
    state_role_map: Dict[str, int],
    state_calibration_stats: Dict[int, Dict[str, float]],
    low_motion_threshold: float,
) -> Dict[str, object]:
    # persist calibration artifacts so inference and visualization share the same role semantics
    max_tau_values = [stats["max_tau_t"] for stats in sequence_stats]
    near_exit_ratios = [stats["near_exit_ratio"] for stats in sequence_stats]
    borderline_dwell_threshold, exit_dwell_threshold = _quantile_thresholds(max_tau_values)
    borderline_near_exit_ratio_threshold, exit_near_exit_ratio_threshold = _quantile_thresholds(
        near_exit_ratios
    )

    return {
        "state_abs_v_mean": state_abs_v_mean,
        "state_role_map": state_role_map,
        "state_calibration_stats": state_calibration_stats,
        "low_motion_threshold": low_motion_threshold,
        "borderline_dwell_threshold": borderline_dwell_threshold,
        "exit_dwell_threshold": exit_dwell_threshold,
        "borderline_near_exit_ratio_threshold": borderline_near_exit_ratio_threshold,
        "exit_near_exit_ratio_threshold": exit_near_exit_ratio_threshold,
    }


def load_data():
    """Load and concatenate all per-video feature CSV files."""
    csv_files = glob.glob(os.path.join(config.FEATURE_FOLDER, "*.csv"))
    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    sequence_stats: List[Dict[str, float]] = []

    if not csv_files:
        raise FileNotFoundError(f"No CSV feature files found in: {config.FEATURE_FOLDER}")

    for file in csv_files:
        sequence, stats = _read_feature_csv(file)
        sequences.append(sequence)
        lengths.append(len(sequence))
        sequence_stats.append(stats)

    x_all = np.vstack(sequences)
    return x_all, lengths, sequence_stats


def _print_training_summary(
    x: np.ndarray,
    lengths: Sequence[int],
    state_abs_v_mean: Dict[int, float],
    state_calibration_stats: Dict[int, Dict[str, float]],
    state_role_map: Dict[str, int],
    low_motion_threshold: float,
    meta: Dict[str, object],
) -> None:
    print("Total frames:", x.shape[0])
    print("Number of videos:", len(lengths))
    print("Trained")
    print("State abs(v) means:", state_abs_v_mean)
    print("State calibration stats:", state_calibration_stats)
    print("State role map:", state_role_map)
    print("Low-motion abs(v) threshold:", low_motion_threshold)
    print("Exit dwell thresholds:", meta["borderline_dwell_threshold"], meta["exit_dwell_threshold"])
    print(
        "Near-exit ratio thresholds:",
        meta["borderline_near_exit_ratio_threshold"],
        meta["exit_near_exit_ratio_threshold"],
    )


def train():
    x, lengths, sequence_stats = load_data()

    # standardize features across all concatenated frames
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # fit hmm on scaled observations with sequence boundaries to preserve per-video temporal structure
    model = hmm.GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="full",
        n_iter=N_ITER,
    )

    model.fit(x_scaled, lengths)

    abs_v = np.abs(x[:, 3])
    # derive role mapping and calibration stats from fitted states then save them as model metadata
    state_abs_v_mean, state_role_map, state_calibration_stats = _calibrate_state_roles(
        model,
        x_scaled,
        x,
    )
    low_motion_threshold = _compute_low_motion_threshold(abs_v, lengths)
    meta = _build_meta(
        sequence_stats,
        state_abs_v_mean,
        state_role_map,
        state_calibration_stats,
        low_motion_threshold,
    )

    # persist artifacts used by inference
    project_root = CURRENT_DIR.parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "hmm_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(meta, models_dir / "hmm_meta.pkl")

    _print_training_summary(
        x,
        lengths,
        state_abs_v_mean,
        state_calibration_stats,
        state_role_map,
        low_motion_threshold,
        meta,
    )


if __name__ == "__main__":
    train()


