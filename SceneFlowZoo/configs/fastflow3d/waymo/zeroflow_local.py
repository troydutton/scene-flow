_base_ = ["../../pseudoimage.py"]

# ---------------------------------------------------------------------------
# Local paths — override the EFS paths from the parent configs
# ---------------------------------------------------------------------------

test_dataset_root = "/data/troy/datasets/waymo_open_processed_flow/validation/"
save_output_folder = "/data/troy/predictions/waymo_zeroflow/"

# Skip re-computing inference for frames where predictions already exist on disk
cache_validation_outputs = True

# No ground-truth flow labels in our preprocessed data (raw tfrecords don't include
# scene flow annotations) — skip the evaluation step entirely
has_labels = False

SEQUENCE_LENGTH = 2

model = dict(
    name="FastFlow3D",
    args=dict(
        VOXEL_SIZE={{_base_.VOXEL_SIZE}},
        PSEUDO_IMAGE_DIMS={{_base_.PSEUDO_IMAGE_DIMS}},
        POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}},
        FEATURE_CHANNELS=32,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
    ),
)

######## TEST DATASET ########

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="WaymoOpenCausalSceneFlow",
        root_dir=test_dataset_root,
        flow_folder=None,
        with_rgb=False,
        eval_type="bucketed_epe",
        max_pc_points=180000,
        allow_pc_slicing=True,
        # Output path for eval metrics (will be skipped since no GT flow labels)
        eval_args=dict(output_path="eval_results/bucketed_epe/waymo/zeroflow_local/"),
    ),
)

test_dataloader = dict(args=dict(batch_size=4, num_workers=4, shuffle=False, pin_memory=True))
