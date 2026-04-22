import argparse
import copy
import numpy as np
import plotly.graph_objects as go
from pcdet.utils import box_utils
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

def draw_scenes_plotly(points, gt_boxes, filename='visualization.html'):
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

    # Match typical LiDAR coordinate orientation
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(filename)
    return filename

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for your model/dataset (e.g. cfgs/waymo_models/pv_rcnn.yaml)')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)

    # Fallback to dataset config if a model config isn't provided
    if not hasattr(cfg, 'DATA_CONFIG'):
        print("Warning: DATA_CONFIG not found in config. Assuming a raw dataset config was provided.")
        cfg.DATA_CONFIG = copy.deepcopy(cfg)
        cfg.CLASS_NAMES = ['Vehicle', 'Pedestrian', 'Cyclist'] # default waymo/kitti classes

    # Disable voxelization to avoid spconv dependency during raw point cloud visualization
    if hasattr(cfg.DATA_CONFIG, 'DATA_PROCESSOR'):
        cfg.DATA_CONFIG.DATA_PROCESSOR = [
            processor for processor in cfg.DATA_CONFIG.DATA_PROCESSOR 
            if processor.NAME != 'transform_points_to_voxels'
        ]

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Ground Truth Visualization -------------------------')

    # Build the dataset using your existing config 
    # Use split='train' or 'val' to read the set carrying ground truth boxes
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

    # Retrieve items directly from the dataset class
    for idx in range(len(dataset)):
        logger.info(f'Visualizing sample index: \t{idx}')
        
        # Pull raw item out of dataset
        data_dict = dataset[idx]
        
        # For open3d visualization, we need (N, 3) points and numpy boxes
        # The points tensor usually has structure [x, y, z, intensity, ...]
        points = data_dict['points'][:, :3] 

        # We pass gt_boxes specifically (ignoring predictions)
        gt_boxes = data_dict['gt_boxes'] 
        
        # Discard the label/class ID from boxes if it happens to be present in dim 7
        if gt_boxes.shape[-1] > 7:
            gt_boxes = gt_boxes[:, :7]

        # Draw point cloud and ground truth boxes into an interactive HTML file
        output_file = f'visualization_{idx}.html'
        draw_scenes_plotly(
            points=points, 
            gt_boxes=gt_boxes,
            filename=output_file
        )
        logger.info(f"Saved visualization to {output_file}")
        
        # We break after one sample so it doesn't overwrite / generate infinitely 
        # Remove this break to generate more samples
        break

if __name__ == '__main__':
    main()
