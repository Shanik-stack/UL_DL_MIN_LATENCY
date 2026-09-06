import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from .simulation import collect_uplink_interference_diagnostics
def to_numpy_safe(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_result_save_dir(save_dir):
    if save_dir is None:
        raise ValueError("save_dir is required for every plot.")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _add_legend_if_present(ax, *, handles=None, labels=None, **kwargs):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if handles and any(str(label).strip() for label in labels):
        ax.legend(handles=handles, labels=labels, **kwargs)


def _extract_uplink_block_results(plot_data):
    return (
        plot_data.get("all_user_block_results")
        or plot_data.get("all_user_block_results_train")
        or plot_data.get("all_user_block_results_test")
        or []
    )


def _extract_uplink_epoch_rate_panel_data(plot_data, user_idx):
    training_history = plot_data.get("precoder_net_training_history", {})
    if isinstance(training_history, dict):
        per_user_rate = training_history.get("per_user_rate", [])
        if user_idx < len(per_user_rate):
            user_rates = [float(v) for v in list(per_user_rate[user_idx]) if np.isfinite(float(v))]
            if len(user_rates) > 0:
                return {
                    "source_label": "Training R_fbl over epoch",
                    "x_label": "Training epoch",
                    "segments": [
                        {
                            "x": np.arange(1, len(user_rates) + 1, dtype=float),
                            "y": np.asarray(user_rates, dtype=float),
                            "label": "training",
                            "color_index": 0,
                        }
                    ],
                    "separators": [],
                }

    all_user_block_results = _extract_uplink_block_results(plot_data)
    if user_idx >= len(all_user_block_results):
        return None

    segments = []
    separators = []
    cursor = 0.0
    for block_idx, block_states in enumerate(all_user_block_results[user_idx] or []):
        for solve_idx, state in enumerate(block_states or []):
            kkt_history = list(state.get("kkt_history", []))
            if len(kkt_history) == 0:
                continue
            x_vals = cursor + np.arange(1, len(kkt_history) + 1, dtype=float)
            y_vals = np.asarray(
                [
                    float(
                        row.get(
                            "rate",
                            state.get("achieved_R_fbl", np.nan),
                        )
                    )
                    for row in kkt_history
                ],
                dtype=float,
            )
            n_val = int(state.get("n_kl", 0))
            segments.append(
                {
                    "x": x_vals,
                    "y": y_vals,
                    "label": f"block {block_idx}, n={n_val}",
                    "color_index": int(block_idx),
                }
            )
            cursor = float(x_vals[-1]) + 1.5
            separators.append(float(x_vals[-1]) + 0.5)

    if len(segments) == 0:
        return None

    return {
        "source_label": "Optimization R_fbl over epoch",
        "x_label": "Epoch within solve",
        "segments": segments,
        "separators": separators[:-1],
    }


def _plot_uplink_rfbl_vs_n_axis(ax, user_blocks, user_idx):
    plotted_any = False
    for block_idx, states in enumerate(user_blocks or []):
        if not states:
            continue

        n_vals, achieved_vals, required_vals = [], [], []
        for item in states:
            if "n_kl" not in item or "R_fbl" not in item:
                continue
            n_val = int(item["n_kl"])
            n_vals.append(n_val)
            achieved_vals.append(float(item.get("achieved_R_fbl", item["R_fbl"])))
            if "required_R_fbl" in item:
                required_vals.append(float(item["required_R_fbl"]))
            elif "target_bits" in item:
                required_vals.append(float(item["target_bits"]) / float(max(n_val, 1)))
            else:
                required_vals.append(float(item.get("Bits per sub-block length B/n_kl", np.nan)))

        if len(n_vals) == 0:
            continue

        order = np.argsort(n_vals)
        n_vals = np.asarray(n_vals, dtype=float)[order]
        achieved_vals = np.asarray(achieved_vals, dtype=float)[order]
        required_vals = np.asarray(required_vals, dtype=float)[order]
        ax.plot(n_vals, achieved_vals, marker="o", label=f"block {block_idx} achieved")
        if np.any(np.isfinite(required_vals)):
            ax.plot(n_vals, required_vals, linestyle="--", alpha=0.65, label=f"block {block_idx} required")
        plotted_any = True

    ax.set_xlabel(r"$n_{k,\ell}$")
    ax.set_ylabel(r"Rate (bits / symbol)")
    ax.set_title(f"User {user_idx}: Achieved vs required rate")
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(fontsize=8, ncol=2)
    ax.invert_xaxis()


def _plot_uplink_epoch_rate_axis(ax, plot_data, user_idx):
    panel_data = _extract_uplink_epoch_rate_panel_data(plot_data, user_idx)
    if panel_data is None:
        ax.set_title(f"User {user_idx}: R_fbl over epoch")
        ax.text(0.5, 0.5, "No epoch-rate history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    cmap = plt.get_cmap("tab20")
    shown_labels = set()
    for segment in panel_data["segments"]:
        label = segment["label"] if segment["label"] not in shown_labels else None
        if label is not None:
            shown_labels.add(label)
        ax.plot(
            segment["x"],
            segment["y"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            alpha=0.9,
            color=cmap(int(segment["color_index"]) % cmap.N),
            label=label,
        )
    for separator_x in panel_data.get("separators", []):
        ax.axvline(separator_x, color="black", linestyle=":", linewidth=0.8, alpha=0.3)

    ax.set_title(f"User {user_idx}: {panel_data['source_label']}")
    ax.set_xlabel(panel_data["x_label"])
    ax.set_ylabel("Achieved R_fbl")
    ax.grid(True, alpha=0.3)
    if len(shown_labels) > 0 and len(shown_labels) <= 8:
        ax.legend(fontsize=8, loc="best")


def _plot_uplink_stacked_rfbl_vs_n_and_epoch(plot_data, save_dir):
    all_user_block_results = _extract_uplink_block_results(plot_data)
    K = len(all_user_block_results)
    if K == 0:
        return

    fig, axes = plt.subplots(K, 2, figsize=(14, max(4 * K, 5)), squeeze=False)
    for user_idx, user_blocks in enumerate(all_user_block_results):
        _plot_uplink_rfbl_vs_n_axis(axes[user_idx, 0], user_blocks, user_idx)
        _plot_uplink_epoch_rate_axis(axes[user_idx, 1], plot_data, user_idx)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "all_users_Rfbl_vs_n_with_epoch.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_optimization_result_summary_dict(
    post_training_data_dict,
    train=True,
    save_dir=None,
    phase_label=None,
    filename_prefix=None,
):
    """
    Uses module global experiment folders unless save_dir is explicitly given.

    Expected keys:
      - training: 'n_star', 'R_star'
      - testing summary remap should also provide same keys

    Saves:
      training_user{user}_summary.png   if train=True
      testing_user{user}_summary.png    if train=False
    """
    save_dir = _get_result_save_dir(save_dir)

    n_star = post_training_data_dict.get("n_star", None)
    R_star = post_training_data_dict.get("R_star", None)
    achieved_R_star = post_training_data_dict.get("achieved_R_star", None)
    required_R_star = post_training_data_dict.get("required_R_star", None)

    if n_star is None or R_star is None:
        raise KeyError("post_training_data_dict must contain keys: 'n_star' and 'R_star'")

    if len(n_star) != len(R_star):
        raise ValueError(
            f"Length mismatch: len(n_star)={len(n_star)} but len(R_star)={len(R_star)}"
        )

    title_prefix = phase_label or ("Training" if train else "Testing")
    file_prefix = filename_prefix or ("training" if train else "testing")

    if achieved_R_star is None or required_R_star is None:
        block_results = (
            post_training_data_dict.get("all_user_block_results")
            or post_training_data_dict.get("all_user_block_results_train")
            or post_training_data_dict.get("all_user_block_results_test")
        )
        if block_results is not None:
            derived_achieved = []
            derived_required = []
            for user_blocks in block_results:
                user_achieved = []
                user_required = []
                for block_states in user_blocks:
                    if block_states is None or len(block_states) == 0:
                        user_achieved.append(np.nan)
                        user_required.append(np.nan)
                        continue
                    final_state = block_states[-1]
                    n_val = float(max(int(final_state.get("n_kl", 1)), 1))
                    served_rate = float(final_state.get("Bits per sub-block length B/n_kl", np.nan))
                    target_bits = final_state.get("target_bits", None)
                    required_rate = (
                        float(final_state.get("required_R_fbl"))
                        if "required_R_fbl" in final_state
                        else (
                            float(target_bits) / n_val
                            if target_bits is not None
                            else served_rate
                        )
                    )
                    achieved_rate = float(final_state.get("achieved_R_fbl", np.nan))
                    user_achieved.append(achieved_rate)
                    user_required.append(required_rate)
                derived_achieved.append(user_achieved)
                derived_required.append(user_required)
            if achieved_R_star is None:
                achieved_R_star = derived_achieved
            if required_R_star is None:
                required_R_star = derived_required

    for user_idx in range(len(n_star)):
        L = len(n_star[user_idx])
        if L == 0:
            continue

        blocks = np.arange(L)
        t_vals = np.asarray(n_star[user_idx], dtype=np.float64)
        achieved_vals = np.asarray(
            (
                achieved_R_star[user_idx]
                if achieved_R_star is not None and user_idx < len(achieved_R_star)
                else R_star[user_idx]
            ),
            dtype=np.float64,
        )
        required_vals = np.asarray(
            (
                required_R_star[user_idx]
                if required_R_star is not None and user_idx < len(required_R_star)
                else R_star[user_idx]
            ),
            dtype=np.float64,
        )

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        ax1.plot(blocks, t_vals, marker="o", label="n_kl")
        ax1.set_xlabel("Block index (l)")
        ax1.set_ylabel("Chosen sub-blocklength n_kl")
        ax1.grid(True)

        ax2.plot(blocks, achieved_vals, marker="s", label="Achieved R_fbl")
        if np.any(np.isfinite(required_vals)):
            ax2.plot(blocks, required_vals, marker="^", linestyle="--", label="Required R_fbl")
        ax2.set_ylabel("Rate (bits / symbol)")

        ax1.set_title(f"{title_prefix} Result – User {user_idx}")

        # combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")

        fig.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"{file_prefix}_user{user_idx}_summary.png"
        )

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_F_vs_n_for_all_subblocks(
    post_training_data_dict,
    save_dir="F_vs_n",   # only subfolder name
    base_dir=None,
):
    """Plot per-user blocklength diagnostics below an explicit result folder."""

    base_dir = _get_result_save_dir(base_dir)

    # append subfolder
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_user_block_results = _extract_uplink_block_results(post_training_data_dict)
    if len(all_user_block_results) == 0:
        return

    for user_idx, user_blocks in enumerate(all_user_block_results):
        if not user_blocks:
            continue

        # ==================================================
        # FIG 1: ||F||^2 vs n
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_power_vals = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue

                n_vals.append(int(item["n_kl"]))

                if "F_power" in item:
                    F_power_vals.append(float(item["F_power"]))
                else:
                    F_arr = to_numpy_safe(item["F"])
                    F_power_vals.append(np.sum(np.abs(F_arr) ** 2))

            if len(n_vals) == 0:
                continue

            order = np.argsort(n_vals)
            n_vals = np.array(n_vals)[order]
            F_power_vals = np.array(F_power_vals)[order]

            plt.plot(n_vals, F_power_vals, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F\|_F^2$")
        plt.title(f"User {user_idx}: Precoder power vs n")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()   # correct position

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fpower_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 2: ||F(n) - F(n_max)|| vs n
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_list = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue
                n_vals.append(int(item["n_kl"]))
                F_list.append(to_numpy_safe(item["F"]))

            if len(n_vals) == 0:
                continue

            order_desc = np.argsort(n_vals)[::-1]
            n_desc = np.array(n_vals)[order_desc]
            F_desc = [F_list[i] for i in order_desc]

            F_ref = F_desc[0]

            delta = [
                np.linalg.norm((F - F_ref).reshape(-1))
                for F in F_desc
            ]

            order_inc = np.argsort(n_desc)
            n_plot = n_desc[order_inc]
            d_plot = np.array(delta)[order_inc]

            plt.plot(n_plot, d_plot, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F(n) - F(n_{\max})\|_F$")
        plt.title(f"User {user_idx}: Precoder drift vs n")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fchange_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 3: incremental change
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_list = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue
                n_vals.append(int(item["n_kl"]))
                F_list.append(to_numpy_safe(item["F"]))

            if len(n_vals) == 0:
                continue

            order_desc = np.argsort(n_vals)[::-1]
            n_desc = np.array(n_vals)[order_desc]
            F_desc = [F_list[i] for i in order_desc]

            delta_prev = [0.0]
            for i in range(1, len(F_desc)):
                delta_prev.append(
                    np.linalg.norm((F_desc[i] - F_desc[i - 1]).reshape(-1))
                )

            order_inc = np.argsort(n_desc)
            n_plot = n_desc[order_inc]
            d_plot = np.array(delta_prev)[order_inc]

            plt.plot(n_plot, d_plot, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F_i - F_{i-1}\|_F$")
        plt.title(f"User {user_idx}: incremental F change")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fincrement_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 4: R_fbl vs n
        # ==================================================
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
        _plot_uplink_rfbl_vs_n_axis(axes[0, 0], user_blocks, user_idx)
        _plot_uplink_epoch_rate_axis(axes[0, 1], post_training_data_dict, user_idx)
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"user{user_idx}_Rfbl_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    _plot_uplink_stacked_rfbl_vs_n_and_epoch(post_training_data_dict, save_dir)

def plot_optimization_result(
    all_user_results,
    train=True,
    save_dir=None,
    phase_label=None,
    filename_prefix=None,
):
    """
    all_user_results[user][block] = list_of_dicts_over_n_kl

    Uses module global experiment folders unless save_dir is explicitly given.

    Saves:
      training_optimization_user{u}_block{b}.png   if train=True
      testing_optimization_user{u}_block{b}.png    if train=False
    """
    save_dir = _get_result_save_dir(save_dir)
    title_prefix = phase_label or ("Training" if train else "Testing")
    file_prefix = filename_prefix or ("training" if train else "testing")

    for user_idx, user_result in enumerate(all_user_results):
        for block_idx, block_result in enumerate(user_result):
            if block_result is None or len(block_result) == 0:
                continue

            t_vals = np.asarray([d["n_kl"] for d in block_result], dtype=np.float64)
            served_rate_vals = np.asarray(
                [d["Bits per sub-block length B/n_kl"] for d in block_result],
                dtype=np.float64
            )
            achieved_rate_vals = np.asarray(
                [d.get("achieved_R_fbl", d["R_fbl"]) for d in block_result],
                dtype=np.float64,
            )
            required_rate_vals = np.asarray(
                [
                    (
                        d["required_R_fbl"]
                        if "required_R_fbl" in d
                        else (
                            float(d["target_bits"]) / float(max(int(d["n_kl"]), 1))
                            if "target_bits" in d
                            else d["Bits per sub-block length B/n_kl"]
                        )
                    )
                    for d in block_result
                ],
                dtype=np.float64,
            )

            fig = plt.figure(figsize=(6, 4))
            plt.plot(t_vals, served_rate_vals, marker="o", label="Served bits / n_kl")
            if np.any(np.isfinite(required_rate_vals)):
                plt.plot(t_vals, required_rate_vals, marker="^", linestyle="--", label="Required R_fbl")
            plt.plot(t_vals, achieved_rate_vals, marker="s", label="Achieved R_fbl")

            t_final = t_vals[-1]
            served_final = served_rate_vals[-1]
            achieved_final = achieved_rate_vals[-1]

            plt.annotate(
                f"served: {served_final:.4f}",
                (t_final, served_final),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"),
            )

            plt.annotate(
                f"achieved: {achieved_final:.4f}",
                (t_final, achieved_final),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"),
            )

            plt.xlabel("Blocklength n_kl")
            plt.ylabel("Rate (bits / symbol)")

            plt.title(f"{title_prefix} Result – User {user_idx}, Block {block_idx}")

            plt.legend()
            plt.grid(True)
            plt.gca().invert_xaxis()
            plt.tight_layout()

            save_path = os.path.join(
                save_dir,
                f"{file_prefix}_optimization_user{user_idx}_block{block_idx}.png"
            )

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_per_user_schedule_details(result, figs_dir):
    K = len(result.get("n_kl", result.get("final_n_kl", [])))
    if K == 0:
        return

    os.makedirs(figs_dir, exist_ok=True)
    initial_n = result.get("initial_n_kl", [[] for _ in range(K)])
    initial_b = result.get("initial_B_kl", [[] for _ in range(K)])
    final_latency = np.asarray(result.get("final_latency", [0.0] * K), dtype=float)
    initial_latency = np.asarray(result.get("initial_latency", [0.0] * K), dtype=float)
    final_n = result.get("n_kl", result.get("final_n_kl", [[] for _ in range(K)]))
    final_b = result.get("B_kl", [[] for _ in range(K)])
    final_rates = result.get("final_R_fbl", result.get("R_fbl", [[] for _ in range(K)]))

    fig, axes = plt.subplots(K, 2, figsize=(14, max(4 * K, 6)), squeeze=False)
    for k in range(K):
        blocks = np.arange(len(final_n[k]))
        bits = np.asarray(final_b[k], dtype=float) if k < len(final_b) else np.asarray([], dtype=float)
        n_vals = np.asarray(final_n[k], dtype=float) if k < len(final_n) else np.asarray([], dtype=float)
        rates = np.asarray(final_rates[k], dtype=float) if k < len(final_rates) else np.asarray([], dtype=float)
        required = bits / np.maximum(n_vals, 1.0)
        margins = rates - required

        init_blocks = np.arange(len(initial_n[k])) if k < len(initial_n) else np.asarray([], dtype=int)
        init_bits = np.asarray(initial_b[k], dtype=float) if k < len(initial_b) else np.asarray([], dtype=float)
        init_n_vals = np.asarray(initial_n[k], dtype=float) if k < len(initial_n) else np.asarray([], dtype=float)

        ax_left = axes[k, 0]
        ax_left.set_title(
            f"User {k} schedule | init={initial_latency[k]:.4e}s final={final_latency[k]:.4e}s"
        )
        if len(init_blocks) > 0:
            ax_left.bar(init_blocks - 0.18, init_bits, width=0.36, alpha=0.35, label="Initial bits")
        if len(blocks) > 0:
            ax_left.bar(blocks + 0.18, bits, width=0.36, alpha=0.8, label="Optimized bits")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Bits")
        ax_left.grid(True, axis="y", alpha=0.3)

        ax_left_r = ax_left.twinx()
        if len(init_blocks) > 0:
            ax_left_r.plot(init_blocks, init_n_vals, "o--", color="tab:orange", alpha=0.6, label="Initial n_kl")
        if len(blocks) > 0:
            ax_left_r.plot(blocks, n_vals, "o-", color="tab:red", label="Optimized n_kl")
        ax_left_r.set_ylabel("Blocklength n_kl")

        handles_l, labels_l = ax_left.get_legend_handles_labels()
        handles_r, labels_r = ax_left_r.get_legend_handles_labels()
        _add_legend_if_present(
            ax_left,
            handles=handles_l + handles_r,
            labels=labels_l + labels_r,
            fontsize=8,
            loc="upper right",
        )

        ax_right = axes[k, 1]
        if len(blocks) > 0:
            ax_right.plot(blocks, rates, "o-", label="Achieved R_fbl", color="tab:blue")
            ax_right.plot(blocks, required, "x--", label="Required rate", color="tab:green")
            ax_right.bar(blocks, margins, alpha=0.25, color="tab:purple", label="Rate margin")
        ax_right.axhline(0.0, color="black", linewidth=1, linestyle=":")
        ax_right.set_title(f"User {k} per-block rate results")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("Rate (bits/channel-use)")
        ax_right.grid(True, alpha=0.3)
        _add_legend_if_present(ax_right, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_schedule_details.png"), dpi=250)
    plt.close(fig)


def plot_user_config(system_params, save_dir, extra_params=None, max_ticks=12):
    """Save all per-user configuration values in one summary figure."""
    save_path = _get_result_save_dir(save_dir)
    K = int(system_params["K"])
    users = np.arange(K)
    params = {
        "Configured SNR (dB)": np.asarray(system_params["snr_db"]),
        "Power budget": np.asarray(system_params["P"]),
        "Bits": np.asarray(system_params["B"]),
        "Maximum blocklength": np.asarray(system_params["T"]),
        "Initial bits per symbol": np.asarray(system_params["initial_bits_per_symbol"]),
    }
    if extra_params is not None:
        extra_labels = {
            "measured_snr_db_k": "Measured SNR (dB)",
            "measured_sinr_db_k": "Measured SINR (dB)",
        }
        for name, values in extra_params.items():
            arr = np.asarray(values)
            if arr.shape != (K,):
                raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
            label = extra_labels.get(str(name), str(name).replace("_", " ").title())
            params[label] = arr

    tick_step = max(1, int(np.ceil(max(K, 1) / max(int(max_ticks), 1))))
    user_ticks = users[::tick_step]
    n_params = len(params)
    ncols = 2
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4 * nrows, 6)), squeeze=False)
    flat_axes = axes.ravel()
    for index, (name, values) in enumerate(params.items()):
        ax = flat_axes[index]
        ax.plot(users, values, marker="o", markersize=5)
        ax.set_title(name)
        ax.set_xlabel("User")
        ax.set_ylabel(name)
        ax.set_xticks(user_ticks)
        ax.grid(True, alpha=0.3)
    for i in range(n_params, len(flat_axes)):
        fig.delaxes(flat_axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "user_config_summary.png"), dpi=250)
    plt.close(fig)

def plot_link_quality_from_json(json_path, save_dir=None, prefix="test"):
    """
    Plot target SNR together with measured SNR/SINR before and after optimization.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_snr_db = np.asarray(data.get("target_snr_db", []), dtype=np.float64)
    initial_snr_db = np.asarray(data.get("initial_snr_db", []), dtype=np.float64)
    final_snr_db = np.asarray(data.get("final_snr_db", []), dtype=np.float64)
    initial_sinr_db = np.asarray(data.get("initial_sinr_db", []), dtype=np.float64)
    final_sinr_db = np.asarray(data.get("final_sinr_db", []), dtype=np.float64)

    if target_snr_db.size == 0 or initial_sinr_db.size == 0 or final_sinr_db.size == 0:
        return

    K = len(target_snr_db)
    user_idx = np.arange(K)
    tick_spacing = 1 if K <= 20 else (5 if K <= 50 else 10)
    sparse_ticks = user_idx[::tick_spacing]
    width = max(10, K * 0.18)

    plt.figure(figsize=(width, 6))
    plt.plot(user_idx, target_snr_db, label="Target SNR (cfg)", linewidth=2)
    if initial_snr_db.size == K:
        plt.plot(user_idx, initial_snr_db, label="Measured SNR (initial)", linestyle="--", alpha=0.85)
    if final_snr_db.size == K:
        plt.plot(user_idx, final_snr_db, label="Measured SNR (final)", linestyle=":", alpha=0.85)
    plt.plot(user_idx, initial_sinr_db, label="Measured SINR (initial)", linewidth=2)
    plt.plot(user_idx, final_sinr_db, label="Measured SINR (final)", linewidth=2)
    plt.xlabel("User Index")
    plt.ylabel("dB")
    plt.title("Per-User Link Quality: Target SNR vs Measured SNR/SINR")
    plt.xticks(sparse_ticks)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_link_quality_comparison.png"), dpi=300)
    plt.close()
def plot_latency_and_asynchronality_from_json(json_path, save_dir=None, prefix="test"):
    """
    Generates a full suite of visualizations for latency and asynchronality:
    1. Bar Comparison 2. Population Hist+CDF 3. Async Distribution 4. Heatmaps
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Data Extraction ---
    initial_latency = np.array(data["initial_latency"], dtype=np.float64)
    final_latency = np.array(data.get("final_latency", data.get("latency")), dtype=np.float64)
    K = len(final_latency)
    user_idx = np.arange(K)

    # --- Adaptive Scaling Logic ---
    dynamic_width = max(10, K * 0.2) 
    tick_spacing = 1 if K <= 20 else (5 if K <= 50 else 10)
    sparse_ticks = user_idx[::tick_spacing]

    # 1) PER-USER BAR COMPARISON
    plt.figure(figsize=(dynamic_width, 6))
    if K > 150:
        plt.plot(user_idx, initial_latency, label="Initial", alpha=0.7, marker='o', markersize=2)
        plt.plot(user_idx, final_latency, label="Final", alpha=0.9, marker='x', markersize=2)
    else:
        bar_width = 0.35 if K <= 50 else 0.3 
        plt.bar(user_idx - bar_width/2, initial_latency, width=bar_width, label="Initial", color='#1f77b4', zorder=3)
        plt.bar(user_idx + bar_width/2, final_latency, width=bar_width, label="Final", color='#ff7f0e', zorder=3)
    plt.xlabel("User Index"); plt.ylabel("Latency"); plt.legend()
    plt.title(f"Per-User Latency Optimization (K={K})")
    plt.xticks(sparse_ticks); plt.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_latency_comparison.png"), dpi=300); plt.close()

    # 2) POPULATION HISTOGRAM + CDF
    fig, ax1 = plt.subplots(figsize=(10, 6))
    combined_data = np.concatenate([initial_latency, final_latency])
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 40)
    ax1.hist(initial_latency, bins=bins, alpha=0.3, color='gray', label="Initial (Freq)")
    ax1.hist(final_latency, bins=bins, alpha=0.5, color='forestgreen', label="Final (Freq)")
    ax1.set_xlabel("Latency Value"); ax1.set_ylabel("Number of Users")
    
    ax2 = ax1.twinx()
    for d, c, ls in zip([initial_latency, final_latency], ['red', 'darkgreen'], ['--', '-']):
        sorted_d = np.sort(d)
        y = np.arange(len(sorted_d)) / float(len(sorted_d) - 1)
        ax2.plot(sorted_d, y, color=c, linestyle=ls, linewidth=2, label=f"CDF {'Init' if ls=='--' else 'Final'}")
    ax2.set_ylabel("Cumulative Probability"); ax2.set_ylim(0, 1.05)
    plt.title("Latency Stats (Hist + CDF)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_latency_histogram.png"), dpi=300); plt.close()

    # 3) ASYNCHRONALITY DISTRIBUTION (Pairs Hist)
    init_diffs = np.abs(initial_latency[:, None] - initial_latency[None, :])
    final_diffs = np.abs(final_latency[:, None] - final_latency[None, :])
    mask = ~np.eye(K, dtype=bool)
    init_vals, final_vals = init_diffs[mask], final_diffs[mask]

    plt.figure(figsize=(10, 6))
    plt.hist(init_vals, bins=50, density=True, alpha=0.4, color='red', label='Initial Async')
    plt.hist(final_vals, bins=50, density=True, alpha=0.6, color='skyblue', label='Final Async')
    plt.title(f"Global Asynchronality Distribution (K={K} users, {len(init_vals)} pairs)")
    plt.xlabel("Latency Difference "); plt.ylabel("Probability Density"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_async_distribution.png"), dpi=300); plt.close()

    # 4) HEATMAPS
    vmin = min(np.min(init_vals), np.min(final_vals))
    vmax = max(np.max(init_vals), np.max(final_vals))
    
    def save_hm(m, title, fname):
        plt.figure(figsize=(max(8, K*0.12), max(6, K*0.1)))
        m_masked = np.ma.masked_where(~mask, m)
        cmap = plt.cm.plasma.copy(); cmap.set_bad(color='white')
        im = plt.imshow(m_masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, label="|Li - Lj|"); plt.xticks(sparse_ticks); plt.yticks(sparse_ticks)
        plt.title(title); plt.xlabel("User j"); plt.ylabel("User i")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=300); plt.close()

    save_hm(init_diffs, "Initial Asynchronality Heatmap", f"{prefix}_initial_async_heatmap.png")
    save_hm(final_diffs, "Final Asynchronality Heatmap", f"{prefix}_final_async_heatmap.png")
    
    
def adapt_training_dict_to_plot_format(post_training_data_dict):
    """
    Your plot_optimization_result_train expects:
        train_all_user_results[user][block] = list_of_dicts_over_n_kl

    The cleaned optimizer returns:
        post_training_data_dict["n_star"][user]   : list of chosen n_kl per block
        post_training_data_dict["R_star"][user]   : list of chosen R_fbl per block
        post_training_data_dict["F_star"][user]   : list of chosen F per block
    and does NOT store intermediate trajectories for each block unless you save them.

    So this adapter builds a minimal structure with ONE point per block:
        list_of_dicts_over_n_kl = [ {final point} ]
    """
    n_star = post_training_data_dict["n_star"]
    R_star = post_training_data_dict["R_star"]
    F_star = post_training_data_dict["F_star"]

    train_all_user_results = []

    K = len(n_star)
    for user in range(K):
        user_blocks = []
        L = len(n_star[user])
        for b in range(L):
            n_kl = int(n_star[user][b])
            R = float(R_star[user][b])
            Fmat = F_star[user][b]

            # Cannot reconstruct B_l reliably unless you saved it during training.
            # Keep it as None; plotting uses "Bits per sub-block length B/n_kl" and "R_fbl".
            # You can set it if you stored B_l in training results.
            point = {
                "n_kl": n_kl,
                "Bits per sub-block length B/n_kl": np.nan,  # fill if you have B_l
                "R_fbl": R,
                "F": Fmat,
            }
            user_blocks.append([point])  # list over "iterations"; here single point
        train_all_user_results.append(user_blocks)

    return train_all_user_results

def _safe_db(values):
    arr = np.asarray(values, dtype=float)
    out = 10.0 * np.log10(np.maximum(arr, 1e-30))
    if np.isscalar(values):
        return float(out)
    return out


def _load_interference_diag(payload):
    if not payload:
        return None
    return {
        "blocks_per_user": [int(v) for v in payload.get("blocks_per_user", [])],
        "signal": np.asarray(payload.get("signal", []), dtype=float),
        "total_interference": np.asarray(payload.get("total_interference", []), dtype=float),
        "noise": np.asarray(payload.get("noise", []), dtype=float),
        "sinr_db": np.asarray(payload.get("sinr_db", []), dtype=float),
        "pairwise_block": np.asarray(payload.get("pairwise_block", []), dtype=float),
        "avg_pairwise_power": np.asarray(payload.get("avg_pairwise_power", []), dtype=float),
        "avg_pairwise_inr_db": np.asarray(payload.get("avg_pairwise_inr_db", []), dtype=float),
        "avg_pairwise_share": np.asarray(payload.get("avg_pairwise_share", []), dtype=float),
        "worst_block": int(payload.get("worst_block", -1)),
    }


def _combined_finite_limits(*matrices):
    finite_parts = []
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return None, None
    stacked = np.concatenate(finite_parts)
    return float(np.min(stacked)), float(np.max(stacked))


def _imshow_with_shared_scale(ax, matrix, title, cbar_label, *, cmap="viridis", center_zero=False, vmin=None, vmax=None):
    mat = np.asarray(matrix, dtype=float)
    ax.set_title(title)
    ax.set_xlabel("Interferer user")
    ax.set_ylabel("Victim user")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    if not np.any(np.isfinite(mat)):
        ax.text(0.5, 0.5, "No active user pair", ha="center", va="center", transform=ax.transAxes)
        return
    masked = np.ma.masked_invalid(mat)
    norm = None
    if center_zero:
        finite = mat[np.isfinite(mat)]
        if finite.size > 0:
            bound = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-12)
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(masked, aspect="auto", interpolation="none", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _mask_inactive_user_pairs(matrix, bits_by_user):
    mat = np.asarray(matrix, dtype=float)
    active = np.asarray(
        [sum(int(bits) for bits in user_bits) > 0 for user_bits in bits_by_user],
        dtype=bool,
    )
    if active.size != mat.shape[0] or mat.shape[0] != mat.shape[1]:
        return mat
    return np.where(active[:, None] & active[None, :], mat, np.nan)


def plot_interference_heatmaps(uplinksystem, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    diag = collect_uplink_interference_diagnostics(uplinksystem)
    avg_power_db = _safe_db(diag["avg_pairwise_power"])
    avg_inr_db = np.asarray(diag["avg_pairwise_inr_db"], dtype=float)
    share_pct = 100.0 * np.asarray(diag["avg_pairwise_share"], dtype=float)
    worst_block = int(diag["worst_block"])
    worst_block_mat = None
    if worst_block >= 0:
        worst_block_mat = _safe_db(np.asarray(diag["pairwise_block"][worst_block], dtype=float))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _imshow_with_shared_scale(axes[0, 0], avg_power_db, "Average pairwise interference power", "dB")
    _imshow_with_shared_scale(axes[0, 1], avg_inr_db, "Average interference-to-noise ratio", "INR (dB)")
    _imshow_with_shared_scale(axes[1, 0], share_pct, "Average interference share", "% of victim interference")
    if worst_block_mat is not None:
        _imshow_with_shared_scale(
            axes[1, 1],
            worst_block_mat,
            f"Worst block interference map (block {worst_block})",
            "dB",
        )
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_interference_before_after_heatmaps(result, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    initial_bits = result.get("initial_B_kl", [])
    final_bits = result.get("B_kl", [])
    initial_power_db = _mask_inactive_user_pairs(
        _safe_db(initial_diag["avg_pairwise_power"]), initial_bits
    )
    final_power_db = _mask_inactive_user_pairs(
        _safe_db(final_diag["avg_pairwise_power"]), final_bits
    )
    delta_power_db = final_power_db - initial_power_db
    initial_inr_db = _mask_inactive_user_pairs(
        np.asarray(initial_diag["avg_pairwise_inr_db"], dtype=float), initial_bits
    )
    final_inr_db = _mask_inactive_user_pairs(
        np.asarray(final_diag["avg_pairwise_inr_db"], dtype=float), final_bits
    )
    delta_inr_db = final_inr_db - initial_inr_db
    power_vmin, power_vmax = _combined_finite_limits(initial_power_db, final_power_db)
    inr_vmin, inr_vmax = _combined_finite_limits(initial_inr_db, final_inr_db)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    _imshow_with_shared_scale(
        axes[0, 0],
        initial_power_db,
        "Avg interference power before optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_shared_scale(
        axes[0, 1],
        final_power_db,
        "Avg interference power after optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_shared_scale(
        axes[0, 2],
        delta_power_db,
        "Interference power change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )
    _imshow_with_shared_scale(
        axes[1, 0],
        initial_inr_db,
        "Avg INR before optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_shared_scale(
        axes[1, 1],
        final_inr_db,
        "Avg INR after optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_shared_scale(
        axes[1, 2],
        delta_inr_db,
        "INR change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_before_after_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_profiles(uplinksystem, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    diag = collect_uplink_interference_diagnostics(uplinksystem)
    signal_db = _safe_db(diag["signal"])
    interference_db = _safe_db(diag["total_interference"])
    noise_db = _safe_db(diag["noise"])
    sinr_db = np.asarray(diag["sinr_db"], dtype=float)
    pairwise_block = np.asarray(diag["pairwise_block"], dtype=float)
    K = int(uplinksystem.K)

    fig, axes = plt.subplots(K, 2, figsize=(15, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        blocks = np.arange(len(uplinksystem.n_kl[k]))
        ax_left = axes[k, 0]
        ax_left.plot(blocks, signal_db[k, : len(blocks)], "o-", label="Signal", color="tab:blue")
        ax_left.plot(blocks, interference_db[k, : len(blocks)], "o-", label="Total interference", color="tab:red")
        ax_left.plot(blocks, noise_db[k, : len(blocks)], "o--", label="Noise", color="tab:gray")
        ax_left.set_title(f"User {k} signal/interference/noise profile")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)

        ax_left_r = ax_left.twinx()
        ax_left_r.plot(blocks, sinr_db[k, : len(blocks)], "s-", color="tab:green", label="SINR")
        ax_left_r.set_ylabel("SINR (dB)")

        handles_l, labels_l = ax_left.get_legend_handles_labels()
        handles_r, labels_r = ax_left_r.get_legend_handles_labels()
        _add_legend_if_present(
            ax_left,
            handles=handles_l + handles_r,
            labels=labels_l + labels_r,
            fontsize=8,
            loc="best",
        )

        ax_right = axes[k, 1]
        contrib = pairwise_block[: len(blocks), k, :]
        bottom = np.zeros(len(blocks), dtype=float)
        for j in range(K):
            if j == k:
                continue
            vals = np.nan_to_num(contrib[:, j], nan=0.0)
            if np.allclose(vals, 0.0):
                continue
            ax_right.bar(blocks, vals, bottom=bottom, alpha=0.75, label=f"Interferer {j}")
            bottom += vals
        ax_right.set_title(f"User {k} interference contributors")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("Interference power")
        ax_right.grid(True, axis="y", alpha=0.3)
        if K <= 6:
            _add_legend_if_present(ax_right, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_profiles.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_before_after(result, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    K = max(
        len(initial_diag.get("blocks_per_user", [])),
        len(final_diag.get("blocks_per_user", [])),
        len(result.get("n_kl", result.get("final_n_kl", []))),
    )
    if K == 0:
        return

    initial_interf_db = _safe_db(initial_diag["total_interference"])
    final_interf_db = _safe_db(final_diag["total_interference"])
    initial_sinr_db = np.asarray(initial_diag["sinr_db"], dtype=float)
    final_sinr_db = np.asarray(final_diag["sinr_db"], dtype=float)
    initial_signal_db = _safe_db(initial_diag["signal"])
    final_signal_db = _safe_db(final_diag["signal"])

    fig, axes = plt.subplots(K, 2, figsize=(16, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        init_blocks = np.arange(int(initial_diag["blocks_per_user"][k])) if k < len(initial_diag["blocks_per_user"]) else np.asarray([], dtype=int)
        final_blocks = np.arange(int(final_diag["blocks_per_user"][k])) if k < len(final_diag["blocks_per_user"]) else np.asarray([], dtype=int)

        ax_left = axes[k, 0]
        if len(init_blocks) > 0:
            ax_left.plot(init_blocks, initial_interf_db[k, : len(init_blocks)], "o--", color="tab:red", alpha=0.8, label="Initial interference")
            ax_left.plot(init_blocks, initial_signal_db[k, : len(init_blocks)], "o--", color="tab:blue", alpha=0.5, label="Initial signal")
        if len(final_blocks) > 0:
            ax_left.plot(final_blocks, final_interf_db[k, : len(final_blocks)], "o-", color="tab:red", label="Final interference")
            ax_left.plot(final_blocks, final_signal_db[k, : len(final_blocks)], "o-", color="tab:blue", alpha=0.7, label="Final signal")
        ax_left.set_title(f"User {k} interference before vs after")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)
        _add_legend_if_present(ax_left, fontsize=8, loc="best")

        ax_right = axes[k, 1]
        if len(init_blocks) > 0:
            ax_right.plot(init_blocks, initial_sinr_db[k, : len(init_blocks)], "s--", color="tab:green", alpha=0.8, label="Initial SINR")
        if len(final_blocks) > 0:
            ax_right.plot(final_blocks, final_sinr_db[k, : len(final_blocks)], "s-", color="tab:green", label="Final SINR")
        ax_right.set_title(f"User {k} SINR before vs after")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("SINR (dB)")
        ax_right.grid(True, alpha=0.3)
        _add_legend_if_present(ax_right, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_before_after.png"), dpi=250)
    plt.close(fig)


def _kkt_status_color(status):
    palette = {
        "kkt_converged": "tab:green",
        "stationary_infeasible": "tab:red",
        "max_epochs_reached": "tab:orange",
        "max_epochs_feasible_best": "tab:blue",
        "max_epochs_best_primal": "tab:purple",
        "unknown": "tab:gray",
        "": "tab:gray",
    }
    return palette.get(str(status), "tab:gray")


def _build_segmented_epoch_positions(rows, *, group_key, gap=1.5):
    if len(rows) == 0:
        return np.asarray([], dtype=float), []

    positions = []
    segments = []
    cursor = 0.0
    idx = 0
    while idx < len(rows):
        group_value = rows[idx].get(group_key)
        segment_start = cursor + 1.0
        local_epoch = 1
        while idx < len(rows) and rows[idx].get(group_key) == group_value:
            positions.append(cursor + float(local_epoch))
            idx += 1
            local_epoch += 1
        segment_length = local_epoch - 1
        segment_end = cursor + float(segment_length)
        segments.append(
            {
                "group": group_value,
                "start": float(segment_start),
                "end": float(segment_end),
                "length": int(segment_length),
            }
        )
        cursor = segment_end + float(gap)
    return np.asarray(positions, dtype=float), segments


def _segmented_epoch_ticks(segments, max_labels_per_segment=4):
    tick_positions = []
    tick_labels = []
    for segment in segments:
        length = int(segment["length"])
        if length <= 0:
            continue
        sample_count = min(max_labels_per_segment, length)
        local_epochs = np.unique(np.linspace(1, length, sample_count, dtype=int))
        for local_epoch in local_epochs:
            tick_positions.append(float(segment["start"]) + float(local_epoch - 1))
            tick_labels.append(str(int(local_epoch)))
    return tick_positions, tick_labels


def _draw_epoch_segment_guides(ax, segments, *, label_prefix, show_segment_labels=False):
    for segment in segments:
        ax.axvline(
            float(segment["start"]) - 0.5,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.25,
        )

    if not show_segment_labels or len(segments) == 0:
        return

    stride = max(1, int(np.ceil(len(segments) / 16.0)))
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for idx, segment in enumerate(segments):
        if idx % stride != 0:
            continue
        center = 0.5 * (float(segment["start"]) + float(segment["end"]))
        ax.text(
            center,
            1.02,
            f"{label_prefix} {segment['group']}",
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )


def plot_kkt_residual_history(
    result,
    train=False,
    save_dir=None,
    phase_label=None,
    filename_prefix=None,
):
    save_dir = _get_result_save_dir(save_dir)
    epoch_history = [
        row for row in list(result.get("epoch_history", []))
        if "kkt_primal_residual" in row
        and "kkt_complementarity_residual" in row
        and "kkt_stationarity_residual" in row
    ]
    if len(epoch_history) == 0:
        return

    tol_cfg = result.get("kkt_tolerances", {})
    residual_specs = [
        ("kkt_primal_residual", "r_p", float(tol_cfg.get("primal", np.nan))),
        ("kkt_complementarity_residual", "r_c", float(tol_cfg.get("complementarity", np.nan))),
        ("kkt_stationarity_residual", "r_s", float(tol_cfg.get("stationarity", np.nan))),
    ]
    file_prefix = filename_prefix or ("training" if train else "testing")
    title_prefix = phase_label or ("Training" if train else "Testing")

    user_ids = sorted({int(row.get("user", 0)) for row in epoch_history})
    for user_id in user_ids:
        user_rows = [row for row in epoch_history if int(row.get("user", 0)) == user_id]
        if len(user_rows) == 0:
            continue

        epoch_positions, segments = _build_segmented_epoch_positions(user_rows, group_key="block")
        status_markers = [
            (float(epoch_positions[idx]), str(row.get("solve_status", "unknown")))
            for idx, row in enumerate(user_rows)
            if bool(row.get("solve_segment_end", False))
        ]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, (field, label, tol_value) in zip(axes, residual_specs):
            values = np.asarray([float(row.get(field, np.nan)) for row in user_rows], dtype=float)
            finite_positive = np.where(np.isfinite(values), np.maximum(values, 1e-12), np.nan)
            ax.semilogy(epoch_positions, finite_positive, color="tab:blue", linewidth=1.5)
            if np.isfinite(tol_value) and tol_value > 0.0:
                ax.axhline(float(tol_value), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            for marker_epoch, status in status_markers:
                ax.axvline(
                    marker_epoch,
                    color=_kkt_status_color(status),
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.35,
                )
            _draw_epoch_segment_guides(
                ax,
                segments,
                label_prefix="Block",
                show_segment_labels=(ax is axes[0]),
            )
            ax.set_ylabel(label)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"User {user_id} {label} residual history")

        tick_positions, tick_labels = _segmented_epoch_ticks(segments)
        axes[-1].set_xticks(tick_positions)
        axes[-1].set_xticklabels(tick_labels)
        axes[-1].set_xlabel("Epoch within block (restarts at each block)")
        present_statuses = []
        for _, status in status_markers:
            if status not in present_statuses:
                present_statuses.append(status)
        legend_handles = [
            Line2D([0], [0], color="tab:blue", linewidth=1.8, label="residual"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tolerance"),
        ]
        for status in present_statuses:
            legend_handles.append(
                Line2D([0], [0], color=_kkt_status_color(status), linestyle=":", linewidth=2.0, label=status)
            )
        axes[0].legend(
            handles=legend_handles,
            fontsize=8,
            loc="upper right",
            ncol=min(3, max(1, len(legend_handles))),
        )
        fig.suptitle(f"{title_prefix} user {user_id} KKT residual convergence history", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        plt.savefig(
            os.path.join(save_dir, f"{file_prefix}_user{user_id}_kkt_residual_history.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
