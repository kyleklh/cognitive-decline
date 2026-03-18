import sys
import matplotlib.pyplot as plt
import numpy as np

from detect_behavior import (
    DEFAULT_ROLE_MAP,
    _compute_role_ratios,
    _compute_role_scores,
    _estimate_dt,
    hmm_meta,
    predict,
)


STATE_LABELS = {
    "normal": "Normal",
    "borderline": "Borderline",
    "exit_seek": "Exit-Seeking",
}


def _build_state_tick_labels():
    # use learned role mapping so axis labels stay correct even when hmm state ids permute across retrains
    role_map = hmm_meta.get("state_role_map", DEFAULT_ROLE_MAP)
    state_to_label = {}
    for role_name, state_id in role_map.items():
        if isinstance(state_id, int):
            state_to_label[state_id] = STATE_LABELS.get(role_name, role_name)
    return dict(sorted(state_to_label.items(), key=lambda item: item[0]))

def visualize(csv_file):

    df = predict(csv_file)

    states = df["state"]
    state_tick_labels = _build_state_tick_labels()

    role_map = hmm_meta.get("state_role_map", DEFAULT_ROLE_MAP)
    role_ratios = _compute_role_ratios(df, role_map)

    total = len(df)
    mean_abs_v = float(df["v_t"].abs().mean())
    max_tau_t = float(df["tau_t"].max())
    near_exit_ratio = float((df["tau_t"] > 0).mean())
    entry_count = int(df["n_t"].max())
    dt_est = _estimate_dt(df)
    clip_duration_seconds = max(total * dt_est, 1e-6)
    approaches_per_minute = float(entry_count * 60.0 / clip_duration_seconds)

    role_scores = _compute_role_scores(
        role_ratios,
        mean_abs_v,
        max_tau_t,
        near_exit_ratio,
        approaches_per_minute,
    )
    predicted_role = max(role_scores, key=role_scores.get)

    role_order = ["normal", "borderline", "exit_seek"]
    role_display = [STATE_LABELS[r] for r in role_order]
    hmm_values = [role_ratios[r] for r in role_order]
    smooth_values = [role_scores[r] for r in role_order]

    # top panel shows frame-by-frame latent states from hmm decoding
    # bottom panel compares normalized hmm occupancy against the smooth-evidence clip score used for final labeling
    fig, (ax_timeline, ax_scores) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1.7]},
    )

    ax_timeline.step(states.index, states.values, where="post", linewidth=2)
    ax_timeline.set_title(
        f"Behavior Timeline + Smooth Evidence | Final: {STATE_LABELS.get(predicted_role, predicted_role)}"
    )
    ax_timeline.set_xlabel("Frame")
    ax_timeline.set_ylabel("State")

    if state_tick_labels:
        ax_timeline.set_yticks(list(state_tick_labels.keys()))
        ax_timeline.set_yticklabels(list(state_tick_labels.values()))

    ax_timeline.grid(True)

    info_text = (
        "Evidence Stats\n"
        f"mean |v_t|={mean_abs_v:.1f}\n"
        f"near-exit={near_exit_ratio:.2f}\n"
        f"max tau_t={max_tau_t:.2f}\n"
        f"approaches/min={approaches_per_minute:.2f}"
    )

    x = np.arange(len(role_order))
    width = 0.38
    ax_scores.bar(x - width / 2, hmm_values, width=width, label="HMM role ratio")
    ax_scores.bar(x + width / 2, smooth_values, width=width, label="Smooth evidence score")
    ax_scores.set_xticks(x)
    ax_scores.set_xticklabels(role_display)
    ax_scores.set_ylabel("Score")
    ax_scores.set_title("Role Comparison: HMM vs Smooth Evidence")
    ax_scores.grid(True, axis="y", alpha=0.3)
    ax_scores.legend()

    # reserve right margin for the evidence summary so text never overlays either subplot
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    fig.subplots_adjust(hspace=0.4, right=0.84)
    fig.text(
        0.98,
        0.88,
        info_text,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/hmm/visualize_hmm.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    visualize(csv_file)