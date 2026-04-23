import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    segment_name = "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"
    pred_dir = f"/data/troy/predictions/waymo_zeroflow/sequence_len_002/{segment_name}"
    gt_dir = f"/data/troy/datasets/waymo_open_processed_flow/validation/{segment_name}"
    
    viz_dir = "/data/troy/predictions/waymo_zeroflow/viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    frame_idx = 10
    
    # 1. Load sample prediction and point cloud
    pred_path = os.path.join(pred_dir, f"{frame_idx:010d}.feather")
    gt_path = os.path.join(gt_dir, f"{frame_idx:06d}.pkl")
    
    print(f"Loading prediction from: {pred_path}")
    df = pd.read_feather(pred_path)
    
    print(f"Loading ground truth from: {gt_path}")
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
        
    pc = gt_data['car_frame_pc']
    
    is_valid = df['is_valid'].values
    flow = df[['flow_tx_m', 'flow_ty_m', 'flow_tz_m']].values
    
    valid_pc = pc[is_valid]
    valid_flow = flow[is_valid]
    
    flow_mag = np.linalg.norm(valid_flow, axis=1)
    
    # 2. Visualize
    fig = plt.figure(figsize=(20, 15))
    
    # a) Top-down XY scatter, colored by flow magnitude
    ax1 = fig.add_subplot(2, 2, 1)
    sc1 = ax1.scatter(valid_pc[:, 0], valid_pc[:, 1], c=flow_mag, cmap='viridis', s=2, alpha=0.8)
    ax1.set_title("Top-down (BEV) valid points colored by flow magnitude")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(sc1, ax=ax1, label='Flow Magnitude (m)')
    ax1.axis('equal')
    
    # b) 3D arrow/quiver plot
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    num_samples = min(500, len(valid_pc))
    sample_indices = np.random.choice(len(valid_pc), num_samples, replace=False)
    
    samp_pc = valid_pc[sample_indices]
    samp_flow = valid_flow[sample_indices]
    
    ax2.quiver(samp_pc[:, 0], samp_pc[:, 1], samp_pc[:, 2],
               samp_flow[:, 0], samp_flow[:, 1], samp_flow[:, 2],
               color='blue', length=1.0, normalize=False, alpha=0.6)
    ax2.set_title("3D Flow Vectors (Random 500 valid points)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    
    # c) Histogram of flow magnitudes
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(flow_mag, bins=50, color='skyblue', edgecolor='black')
    ax3.set_title("Histogram of Flow Magnitudes (Valid points)")
    ax3.set_xlabel("Flow Magnitude (m)")
    ax3.set_ylabel("Count")
    
    # d) Comparison panel: valid vs invalid points
    ax4 = fig.add_subplot(2, 2, 4)
    # Plot invalid first, then valid on top
    invalid_pc = pc[~is_valid]
    ax4.scatter(invalid_pc[:, 0], invalid_pc[:, 1], c='red', s=1, alpha=0.5, label='Invalid (False)')
    ax4.scatter(valid_pc[:, 0], valid_pc[:, 1], c='green', s=1, alpha=0.5, label='Valid (True)')
    ax4.set_title(f"Valid vs Invalid Predictions (Valid fraction: {is_valid.mean():.3f})")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.legend()
    ax4.axis('equal')
    
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, "zeroflow_viz.png")
    plt.savefig(viz_path, dpi=300)
    print(f"Saved visualization to: {viz_path}")
    
    # 4. Aggregate statistics across all frames
    print(f"\nComputing aggregate statistics for segment: {segment_name}")
    
    feather_files = sorted(glob.glob(os.path.join(pred_dir, "*.feather")))
    
    all_valid_flow_mags = []
    valid_fractions = []
    all_angles = []
    
    for f_path in feather_files:
        df = pd.read_feather(f_path)
        is_val = df['is_valid'].values
        valid_fractions.append(is_val.mean())
        
        v_flow = df[['flow_tx_m', 'flow_ty_m', 'flow_tz_m']].values[is_val]
        if len(v_flow) > 0:
            mags = np.linalg.norm(v_flow, axis=1)
            all_valid_flow_mags.extend(mags)
            
            # Flow direction angle in XY plane
            angles = np.arctan2(v_flow[:, 1], v_flow[:, 0])
            all_angles.extend(angles)
            
    all_valid_flow_mags = np.array(all_valid_flow_mags)
    valid_fractions = np.array(valid_fractions)
    all_angles = np.array(all_angles)
    
    print(f"Frames processed: {len(feather_files)}")
    print(f"Mean flow magnitude (valid): {np.mean(all_valid_flow_mags):.4f} m")
    print(f"99th percentile flow magnitude: {np.percentile(all_valid_flow_mags, 99):.4f} m")
    print(f"Fraction of valid points/frame: {np.mean(valid_fractions):.4f} ± {np.std(valid_fractions):.4f}")
    print(f"Mean flow direction angle (XY): {np.mean(all_angles):.4f} radians")

if __name__ == '__main__':
    main()
