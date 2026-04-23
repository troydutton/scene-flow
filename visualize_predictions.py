import os
import glob
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def draw_scenes_plotly(points, valid_mask, flow_vectors, filename='visualization.html'):
    fig = go.Figure()
    
    # Downsample points for responsiveness
    max_pts = 75000
    if len(points) > max_pts:
        indices = np.random.choice(len(points), max_pts, replace=False)
        points_render = points[indices]
        valid_mask_render = valid_mask[indices]
        flow_render = flow_vectors[indices]
    else:
        points_render = points
        valid_mask_render = valid_mask
        flow_render = flow_vectors

    # Separate valid and invalid points for coloring
    invalid_points = points_render[~valid_mask_render]
    valid_points = points_render[valid_mask_render]
    
    fig.add_trace(go.Scatter3d(
        x=invalid_points[:, 0],
        y=invalid_points[:, 1],
        z=invalid_points[:, 2],
        mode='markers',
        marker=dict(size=1.0, color='gray', opacity=0.8),
        name='Invalid Points'
    ))

    # Calculate magnitudes
    valid_flow = flow_render[valid_mask_render]
    flow_magnitudes = np.linalg.norm(valid_flow, axis=1)

    # Plot valid points colored by magnitude
    fig.add_trace(go.Scatter3d(
        x=valid_points[:, 0],
        y=valid_points[:, 1],
        z=valid_points[:, 2],
        mode='markers',
        marker=dict(size=1.0, color=flow_magnitudes, colorscale='Viridis', opacity=0.8, showscale=False),
        name='Valid Points'
    ))

    # Add scene flow lines
    if len(valid_points) > 0:
        render_limit = 5000
        if len(valid_points) > render_limit:
            # Select points with largest magnitudes
            sorted_valid_idx = np.argsort(flow_magnitudes)[::-1]
            valid_idx = sorted_valid_idx[:render_limit]
        else:
            valid_idx = np.arange(len(valid_points))

        x_lines = []
        y_lines = []
        z_lines = []
        colors = []
        
        for i in valid_idx:
            x0, y0, z0 = valid_points[i, 0], valid_points[i, 1], valid_points[i, 2]
            
            # Use raw flow directly (meters per frame)
            dx, dy, dz = valid_flow[i, 0], valid_flow[i, 1], valid_flow[i, 2]
            
            x_lines.extend([x0, x0 + dx, None])
            y_lines.extend([y0, y0 + dy, None])
            z_lines.extend([z0, z0 + dz, None])
            colors.extend([flow_magnitudes[i], flow_magnitudes[i], flow_magnitudes[i]])

        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(
                color=colors,
                colorscale='Jet',
                width=4,
                showscale=True
            ),
            name='Scene Flow'
        ))

    # Match typical LiDAR coordinate orientation
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="Scene Flow Predictions"
    )

    fig.write_html(filename)
    return filename


def main():
    segment_name = "segment-11616035176233595745_3548_820_3568_820_with_camera_labels"
    pred_dir = f"/data/troy/predictions/waymo_zeroflow/sequence_len_002/{segment_name}"
    gt_dir = f"/data/troy/datasets/waymo_open_processed_flow/validation/{segment_name}"
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 60
    
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
    
    output_file = os.path.join(output_dir, "zeroflow_plotly_viz.html")
    
    draw_scenes_plotly(
        points=pc,
        valid_mask=is_valid,
        flow_vectors=flow,
        filename=output_file
    )
    
    print(f"Saved Plotly visualization to: {output_file}")

if __name__ == '__main__':
    main()
