import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from kinetic_energy import KineticEnergy


def pick_csv_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select your CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    if not path:
        print("No file selected. Exiting.")
        exit(0)
    print(f"Selected: {path}\n")
    return path



def load_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    filename = csv_path.replace("\\", "/").split("/")[-1]
    print(f"Loaded  : {filename}")
    print(f"Shape   : {df.shape}  ({df.shape[0]} frames, {df.shape[1]} columns)")

    # Skip non-numeric meta columns
    skip = 0
    for i in range(df.shape[1]):
        if pd.to_numeric(df.iloc[:, i], errors="coerce").isna().any():
            skip = i + 1
    skip = max(skip, 3)

    time_col = df.iloc[:, 1].values.astype(float)

    # Position mm → m
    pos_raw  = df.iloc[:, skip:].values.astype(float) / 1000.0
    n_frames = pos_raw.shape[0]
    n_joints = pos_raw.shape[1] // 3
    labels   = [f"joint_{i}" for i in range(n_joints)]

    pos = pos_raw[:, :n_joints * 3].reshape(n_frames, n_joints, 3)

    # Velocity via finite difference
    dt  = np.diff(time_col)
    dt  = np.where(dt == 0, 1e-6, dt)
    vel = np.diff(pos, axis=0) / dt[:, None, None]
    # Drop last frame (padded duplicate) — avoids end spike
    vel = vel[:-1]
    time_col = time_col[:-1]

    # Clip outlier spikes at 99th percentile
    p99 = np.percentile(np.abs(vel), 99)
    vel = np.clip(vel, -p99, p99)

    print(f"Joints  : {n_joints}  (auto-detected)")
    print(f"Frames  : {len(time_col)}")
    print(f"Time    : {time_col[0]:.2f}s → {time_col[-1]:.2f}s\n")

    return vel, labels, time_col



def run_real_test(csv_path):
    vel, labels, time_col = load_csv(csv_path)
    n_frames, n_joints, _ = vel.shape

    masses = [1.0] * n_joints
    ke     = KineticEnergy(weights=masses, labels=labels)

    total_ke     = np.zeros(n_frames)
    joint_ke     = {j: np.zeros(n_frames) for j in labels}
    component_ke = np.zeros((n_frames, 3))

    for t in range(n_frames):
        result          = ke(vel[t])
        total_ke[t]     = result["total_energy"]
        component_ke[t] = result["component_energy"]
        for j in labels:
            joint_ke[j][t] = result["joints"][j]["total"]

    # Find top 5 most active joints for readable panel 2
    mean_per_joint = np.array([joint_ke[j].mean() for j in labels])
    top5_idx       = np.argsort(mean_per_joint)[::-1][:5]
    top5_labels    = [labels[i] for i in top5_idx]

    print("=== Real-Data Test PASSED ===")
    print(f"  Frames        : {n_frames}")
    print(f"  Joints        : {n_joints}")
    print(f"  Mean total KE : {total_ke.mean():.4f} J")
    print(f"  Peak total KE : {total_ke.max():.4f} J  @ frame {total_ke.argmax()}")
    print(f"  Min  total KE : {total_ke.min():.4f} J  @ frame {total_ke.argmin()}")
    print(f"  Top 5 joints  : {top5_labels}")

    return total_ke, joint_ke, component_ke, labels, top5_labels, time_col



def plot_real(total_ke, joint_ke, component_ke, labels, top5_labels, time_col):
    t    = time_col[:len(total_ke)]
    cmap = plt.colormaps.get_cmap("tab10")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Real-Data Test — KineticEnergy over Motion Sequence",
                 fontweight="bold", fontsize=13)

    # Panel 1: Total KE — smoothed for readability
    from numpy.lib.stride_tricks import sliding_window_view
    window  = 20
    ke_smooth = np.convolve(total_ke, np.ones(window)/window, mode="same")

    axes[0].plot(t, total_ke,  color="steelblue", linewidth=0.6,
                 alpha=0.4, label="Raw")
    axes[0].plot(t, ke_smooth, color="steelblue", linewidth=1.8,
                 label=f"Smoothed (w={window})")
    axes[0].axhline(total_ke.mean(), color="navy", linestyle="--",
                    linewidth=1.2, label=f"Mean = {total_ke.mean():.4f} J")
    axes[0].set_ylabel("KE (J)")
    axes[0].set_title("Total Kinetic Energy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # ── Panel 2: Top 5 most active joints only — readable
    colors5 = ["steelblue", "tomato", "seagreen", "goldenrod", "mediumpurple"]
    for i, joint in enumerate(top5_labels):
        ke_j      = joint_ke[joint]
        ke_j_sm   = np.convolve(ke_j, np.ones(window)/window, mode="same")
        axes[1].plot(t, ke_j_sm, color=colors5[i],
                     linewidth=1.6, label=joint)
    axes[1].set_ylabel("KE (J)")
    axes[1].set_title(f"Top 5 Most Active Joints (smoothed) — out of {len(labels)}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # ── Panel 3: Component-wise KE — smoothed
    axis_labels = ["X (sagittal)", "Y (frontal)", "Z (vertical)"]
    axis_colors = ["crimson", "mediumorchid", "darkcyan"]
    for i, (lbl, col) in enumerate(zip(axis_labels, axis_colors)):
        ck      = component_ke[:, i]
        ck_sm   = np.convolve(ck, np.ones(window)/window, mode="same")
        axes[2].plot(t, ck_sm, label=lbl, color=col, linewidth=1.6)
    axes[2].set_ylabel("KE (J)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Component-Wise Kinetic Energy — X / Y / Z (smoothed)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("real_data_test_results.png", dpi=150)
    plt.show()
    print("\nPlot saved → real_data_test_results.png")



if __name__ == "__main__":
    csv_path = pick_csv_file()
    total_ke, joint_ke, component_ke, labels, top5_labels, time_col = run_real_test(csv_path)
    plot_real(total_ke, joint_ke, component_ke, labels, top5_labels, time_col)