import numpy as np
import matplotlib.pyplot as plt
from kinetic_energy import KineticEnergy



def run_synthetic_test():
    labels = ["hip", "knee", "ankle"]
    masses = [10.0, 5.0, 2.0]
    ke     = KineticEnergy(weights=masses, labels=labels)

    # Analytically known input → verifiable ground truth
    v = np.array([
        [1.0, 0.0, 0.0],   # hip:   0.5 * 10 * 1² =  5.0 J
        [0.0, 2.0, 0.0],   # knee:  0.5 *  5 * 4  = 10.0 J
        [0.0, 0.0, 3.0],   # ankle: 0.5 *  2 * 9  =  9.0 J
    ])                       # expected total       = 24.0 J

    result = ke(v)

    assert np.isclose(result["total_energy"], 24.0), \
        f"Total KE mismatch: expected 24.0 J, got {result['total_energy']:.4f} J"
    assert np.isclose(result["joints"]["hip"]["total"],    5.0)
    assert np.isclose(result["joints"]["knee"]["total"],  10.0)
    assert np.isclose(result["joints"]["ankle"]["total"],  9.0)

    print("=== Static Assertion PASSED ===")
    print(f"  Total KE : {result['total_energy']:.4f} J  (expected 24.0 J)")
    for joint, vals in result["joints"].items():
        print(f"  {joint:>6s} → {vals['total']:.4f} J | components {np.round(vals['components'], 4)}")

    return result



def generate_synthetic_sequence(n_frames=200, seed=42):
    """
    Generates a synthetic velocity sequence with known physical behaviour:
    - hip    : sinusoidal motion on X (walking cycle)
    - knee   : sinusoidal motion on X, phase-shifted
    - ankle  : sinusoidal motion on X + Z (push-off component)
    All amplitudes are analytically defined so KE peaks are predictable.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, n_frames)   # two full gait cycles

    vel = np.zeros((n_frames, 3, 3))             # (frames, joints, xyz)

    # hip: dominant X motion
    vel[:, 0, 0] = 1.5 * np.sin(t)
    vel[:, 0, 1] = 0.2 * np.sin(2 * t)

    # knee: phase-shifted X + small Z
    vel[:, 1, 0] = 1.2 * np.sin(t + np.pi / 4)
    vel[:, 1, 2] = 0.4 * np.cos(t)

    # ankle: push-off pattern — strong Z at toe-off
    vel[:, 2, 0] = 0.8 * np.sin(t + np.pi / 2)
    vel[:, 2, 2] = 0.9 * np.abs(np.sin(t))      # only positive (upward push)

    # Add small Gaussian noise to simulate measurement noise
    vel += rng.normal(0, 0.02, vel.shape)

    return vel, t


def run_sequence_test():
    labels = ["hip", "knee", "ankle"]
    masses = [10.0, 5.0, 2.0]
    ke     = KineticEnergy(weights=masses, labels=labels)

    vel, t = generate_synthetic_sequence(n_frames=200)
    n_frames = vel.shape[0]

    total_ke     = np.zeros(n_frames)
    joint_ke     = {j: np.zeros(n_frames) for j in labels}
    component_ke = np.zeros((n_frames, 3))

    for i in range(n_frames):
        result          = ke(vel[i])
        total_ke[i]     = result["total_energy"]
        component_ke[i] = result["component_energy"]
        for j in labels:
            joint_ke[j][i] = result["joints"][j]["total"]

    print("\n=== Sequence Test PASSED ===")
    print(f"  Frames        : {n_frames}")
    print(f"  Mean total KE : {total_ke.mean():.4f} J")
    print(f"  Peak total KE : {total_ke.max():.4f} J  @ frame {total_ke.argmax()}")
    print(f"  Min  total KE : {total_ke.min():.4f} J  @ frame {total_ke.argmin()}")

    return total_ke, joint_ke, component_ke, t



def plot_static(result):
    """Bar chart: computed vs expected for the static ground truth frame."""
    labels   = list(result["joints"].keys())
    ke_vals  = [result["joints"][j]["total"] for j in labels]
    expected = [5.0, 10.0, 9.0]

    x     = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Synthetic Test — Static Ground Truth Validation",
                 fontweight="bold")

    # Computed vs expected
    axes[0].bar(x - width/2, ke_vals,  width, label="Computed", color="steelblue")
    axes[0].bar(x + width/2, expected, width, label="Expected",
                color="tomato", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Kinetic Energy (J)")
    axes[0].set_title("Per-Joint KE: Computed vs Expected")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Component stacked bar
    comp_data   = np.array([result["joints"][j]["components"] for j in labels])
    axis_labels = ["X", "Y", "Z"]
    colors      = ["crimson", "mediumorchid", "darkcyan"]
    bottom      = np.zeros(len(labels))
    for i, (ax_lbl, col) in enumerate(zip(axis_labels, colors)):
        axes[1].bar(x, comp_data[:, i], width=0.5,
                    bottom=bottom, label=f"{ax_lbl}-axis", color=col)
        bottom += comp_data[:, i]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Kinetic Energy (J)")
    axes[1].set_title("Component-Wise KE per Joint (Stacked)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("synthetic_static_results.png", dpi=150)
    plt.show()
    print("Plot saved → synthetic_static_results.png")


def plot_sequence(total_ke, joint_ke, component_ke, t):
    """Time-series plots for the synthetic gait sequence."""
    colors = {"hip": "steelblue", "knee": "tomato", "ankle": "seagreen"}

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Synthetic Test — KineticEnergy over Gait Sequence",
                 fontweight="bold", fontsize=13)

    # Panel 1 — Total KE
    axes[0].plot(t, total_ke, color="steelblue", linewidth=1.8)
    axes[0].fill_between(t, total_ke, alpha=0.15, color="steelblue")
    axes[0].axhline(total_ke.mean(), color="navy", linestyle="--",
                    linewidth=1.2, label=f"Mean = {total_ke.mean():.4f} J")
    axes[0].set_ylabel("KE (J)")
    axes[0].set_title("Total Kinetic Energy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2 — Per-joint KE
    for joint, ke_vals in joint_ke.items():
        axes[1].plot(t, ke_vals, label=joint,
                     color=colors[joint], linewidth=1.8)
    axes[1].set_ylabel("KE (J)")
    axes[1].set_title("Per-Joint Kinetic Energy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3 — Component-wise KE
    axis_labels = ["X (sagittal)", "Y (frontal)", "Z (vertical)"]
    axis_colors = ["crimson", "mediumorchid", "darkcyan"]
    for i, (lbl, col) in enumerate(zip(axis_labels, axis_colors)):
        axes[2].plot(t, component_ke[:, i], label=lbl,
                     color=col, linewidth=1.5)
    axes[2].set_ylabel("KE (J)")
    axes[2].set_xlabel("Time (rad — gait cycle)")
    axes[2].set_title("Component-Wise Kinetic Energy (X / Y / Z)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("synthetic_sequence_results.png", dpi=150)
    plt.show()
    print("Plot saved → synthetic_sequence_results.png")




if __name__ == "__main__":
    # Step 1: static ground truth assertion + bar chart
    result = run_synthetic_test()
    plot_static(result)

    # Step 2: multi-frame gait sequence + time-series plots
    total_ke, joint_ke, component_ke, t = run_sequence_test()
    plot_sequence(total_ke, joint_ke, component_ke, t)
