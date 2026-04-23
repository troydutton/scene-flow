import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add src to path to import rigid_flow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rigid_flow.aggregation.rigid_aggregation import compute_rigid_flow
from rigid_flow.core.types import BoundingBox

def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # Generate points representing an object
    np.random.seed(42)
    num_points = 20
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    z = np.zeros(num_points)
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    # True rigid motion (Translation + Rotation)
    # Translation: (3, 2, 0)
    # Rotation: 15 degrees around z-axis
    theta = np.radians(25)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    translation = np.array([3.0, 2.0, 0.0])

    true_target = (R @ points.T).T + translation
    raw_flow = (true_target - points).astype(np.float32)

    # Remove Gaussian noise so normal flow points are perfectly correct
    # raw_flow += np.random.normal(0, 0.1, raw_flow.shape).astype(np.float32)

    # Add some large outliers to make robust methods (median) differ from mean
    outlier_indices = np.random.choice(num_points, 5, replace=False)
    random_offsets = np.random.uniform(-4.0, 4.0, size=(len(outlier_indices), 3)).astype(np.float32)
    random_offsets[:, 2] = 0.0  # Keep z=0 for 2D visualization
    raw_flow[outlier_indices] += random_offsets

    # Make a bounding box
    box = BoundingBox(
        center=np.array([0, 0, 0], dtype=np.float32),
        dimensions=np.array([2, 2, 2], dtype=np.float32),
        heading=0.0,
        class_label=1,
        tracking_id="test_box",
        velocity=np.zeros(2, dtype=np.float32)
    )

    point_to_box = np.zeros(num_points, dtype=np.int32)
    boxes = [box]

    methods = ["none", "mean", "median", "weighted_median", "geometric_median", "svd"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        res = compute_rigid_flow(points, raw_flow, point_to_box, boxes, method=method)
        agg_flow = res.flow
        
        # Plot original points
        ax.scatter(points[:, 0], points[:, 1], c='blue', label='Points at t0', s=30, zorder=3)
        
        # Plot the bounding box outline for reference at t0
        rect_x = [-1, 1, 1, -1, -1]
        rect_y = [-1, -1, 1, 1, -1]
        ax.plot(rect_x, rect_y, color='blue', linestyle='--', alpha=0.5)

        # Plot flow vectors
        for j in range(num_points):
            # highlight outliers
            is_outlier = j in outlier_indices
            color = 'red' if is_outlier else 'green'
            alpha = 0.8 if is_outlier else 0.4
            
            # Use quiver for better arrows
            ax.arrow(points[j, 0], points[j, 1], agg_flow[j, 0], agg_flow[j, 1],
                     head_width=0.15, head_length=0.2, fc=color, ec=color, 
                     alpha=alpha, length_includes_head=True, zorder=2)
                     
        # Plot the true target points at t1
        ax.scatter(true_target[:, 0], true_target[:, 1], c='purple', label='True Target', s=30, marker='X', zorder=3)

        method_name = "SVD" if method == "svd" else method.replace('_', ' ').title()
        ax.set_title(method_name, fontsize=14, pad=10)
        ax.set_xlim(-3, 8)
        ax.set_ylim(-3, 9)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Source Points'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='purple', markersize=8, label='True Target Points'),
                Line2D([0], [0], color='green', lw=2, label='Normal Flow'),
                Line2D([0], [0], color='red', lw=2, label='Outlier Flow')
            ]
            ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.02, r"$\mathbf{Figure\ 1.}$ Visualization of rigid body flow aggregation methods.", ha="center", fontsize=18)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../aggregation_2d_viz.png'))
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
