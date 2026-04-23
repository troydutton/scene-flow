import argparse
import copy
import numpy as np
import plotly.graph_objects as go
from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
import imageio
import os
import glob

def draw_scenes_plotly(points, gt_boxes):
    fig = go.Figure()
    
    # Optionally downsample points for responsiveness
    max_pts = 75000
    if len(points) > max_pts:
        indices = np.random.choice(len(points), max_pts, replace=False)
        points_render = points[indices]
    else:
        points_render = points

    fig.add_trace(go.Scatter3d(
        x=points_render[:, 0],
        y=points_render[:, 1],
        z=points_render[:, 2],
        mode='markers',
        marker=dict(size=1.0, color='gray', opacity=0.8),
        name='Points'
    ))

    if gt_boxes is not None and len(gt_boxes) > 0:
        corners = box_utils.boxes_to_corners_3d(gt_boxes)
        
        lines_x = []
        lines_y = []
        lines_z = []
        for box in corners:
            for start, end in [(0,1), (1,2), (2,3), (3,0),
                               (4,5), (5,6), (6,7), (7,4),
                               (0,4), (1,5), (2,6), (3,7)]:
                lines_x.extend([box[start, 0], box[end, 0], None])
                lines_y.extend([box[start, 1], box[end, 1], None])
                lines_z.extend([box[start, 2], box[end, 2], None])

        fig.add_trace(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Bounding Boxes'
        ))

    # Fix camera angle for consistent frames
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=-1.5, z=1.5)
    )

    # Match typical LiDAR coordinate orientation with fixed axis ranges
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[-75, 75]),
            yaxis=dict(title='Y', range=[-75, 75]),
            zaxis=dict(title='Z', range=[-5, 15]),
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    # Render Scene Flow Vectors if they exist
    if 'flow' in globals() or 'flow_vectors' in locals() or ('gt_boxes' in locals() and gt_boxes.shape[-1] >= 9):
        logger = common_utils.create_logger()
        logger.info("Computing rigid scene flow from box velocities...")
        
        flow_colors = []
        flow_magnitudes = []
        
        # Compute pointwise flow based on bbox vx, vy velocity vectors
        if gt_boxes is not None and gt_boxes.shape[-1] >= 9:
            point_flow = np.zeros((points_render.shape[0], 3))
            
            # Find points in boxes to apply foreground flow
            boxes_tensor = torch.from_numpy(gt_boxes[:,:7]).float()
            points_tensor = torch.from_numpy(points_render[:,:3]).unsqueeze(0).float()
            
            points_in_boxes = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor.squeeze(0), boxes_tensor).numpy()
            
            for i in range(gt_boxes.shape[0]):
                mask = (points_in_boxes[i, :] > 0)
                if np.sum(mask) > 0:
                    vx, vy = gt_boxes[i, 7], gt_boxes[i, 8]
                    point_flow[mask, 0] = vx
                    point_flow[mask, 1] = vy
                    point_flow[mask, 2] = 0.0 
                    
            flow_magnitudes = np.linalg.norm(point_flow, axis=1)
            
            valid_flow = flow_magnitudes > 0.0 
            if np.sum(valid_flow) > 0:
                render_limit = 5000
                valid_idx = np.where(valid_flow)[0]
                if len(valid_idx) > render_limit:
                    sorted_valid_idx = np.argsort(flow_magnitudes[valid_idx])[::-1]
                    valid_idx = valid_idx[sorted_valid_idx[:render_limit]]

                x_lines = []
                y_lines = []
                z_lines = []
                colors = []
                for i in valid_idx:
                    x0, y0, z0 = points_render[i, 0], points_render[i, 1], points_render[i, 2]
                    
                    dt = 0.1
                    dx, dy, dz = point_flow[i, 0] * dt, point_flow[i, 1] * dt, point_flow[i, 2] * dt
                    
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

    return fig

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for your model/dataset (e.g. cfgs/waymo_models/pv_rcnn.yaml)')
    parser.add_argument('--start_idx', type=int, default=0, help='start index in the dataset')
    parser.add_argument('--num_frames', type=int, default=10, help='number of frames to process')
    parser.add_argument('--output_gif', type=str, default='scene_flow.gif', help='output gif filename')
    parser.add_argument('--fps', type=int, default=5, help='frames per second for the gif')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)

    if not hasattr(cfg, 'DATA_CONFIG'):
        print("Warning: DATA_CONFIG not found in config. Assuming a raw dataset config was provided.")
        cfg.DATA_CONFIG = copy.deepcopy(cfg)
        cfg.CLASS_NAMES = ['Vehicle', 'Pedestrian', 'Cyclist']

    if hasattr(cfg.DATA_CONFIG, 'SAMPLED_INTERVAL'):
        print("Forcing speed generation for Scene Flow.")
        cfg.DATA_CONFIG.TRAIN_WITH_SPEED = True

    if hasattr(cfg.DATA_CONFIG, 'DATA_PROCESSOR'):
        cfg.DATA_CONFIG.DATA_PROCESSOR = [
            processor for processor in cfg.DATA_CONFIG.DATA_PROCESSOR 
            if processor.NAME != 'transform_points_to_voxels'
        ]

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Ground Truth GIF Visualization -------------------------')

    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=1,
        logger=logger,
        training=True
    )
    
    logger.info(f'Total number of samples: \t{len(dataset)}')

    temp_files = []
    
    start_idx = args.start_idx
    end_idx = min(start_idx + args.num_frames, len(dataset))
    
    for idx in range(start_idx, end_idx):
        logger.info(f'Processing sample index: \t{idx} ({idx - start_idx + 1}/{end_idx - start_idx})')
        
        data_dict = dataset[idx]
        points = data_dict['points'][:, :3] 
        gt_boxes = data_dict['gt_boxes'] 
        
        if gt_boxes.shape[-1] > 9:
            gt_boxes = gt_boxes[:, :9]

        fig = draw_scenes_plotly(points, gt_boxes)
        
        # Save frame to image
        temp_file = f'temp_frame_{idx}.png'
        # kaleido is required for this to work headlessly
        fig.write_image(temp_file, width=800, height=600, scale=2)
        temp_files.append(temp_file)
        
        logger.info(f"Saved temporary frame {temp_file}")
        
    logger.info(f"Compiling {len(temp_files)} frames into {args.output_gif}...")
    
    # Compile GIF
    frames = []
    for temp_file in temp_files:
        frames.append(imageio.v2.imread(temp_file))
        
    imageio.mimsave(args.output_gif, frames, fps=args.fps)
    logger.info(f"Successfully saved GIF to {args.output_gif}")
    
    # Cleanup temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    main()
