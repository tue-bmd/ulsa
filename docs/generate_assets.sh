python active_sampling_temporal.py \
    --agent_config configs/echonet_3_frames.yaml \
    --target_sequence /mnt/USBMD_datasets/echonet_legacy/val/0X10A5FC19152B50A5.hdf5 \
    --image_range -60 0 \
    --precision mixed_float16 \
    --save_dir output/assets_for_docs \
    --override_config '{"downstream_task": null, "io_config": {"gif_fps": 30, "frame_cutoff": 100, "plot_frames_for_presentation_kwargs": {"file_type": "webm"}}, "action_selection": {"num_lines_to_sample": 7}}'