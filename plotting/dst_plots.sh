## TIG

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 1}}'

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 3}}'

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 5}}'


## DST

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames_downstream_task.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 1}}'

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames_downstream_task.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 3}}'

python /ulsa/active_sampling_temporal.py \
    --agent_config=/ulsa/configs/echonetlvh_3_frames_downstream_task.yaml \
    --target_sequence={data_root}/USBMD_datasets/echonetlvh/val/0X1A00ECE7F179F8ED.hdf5 \
    --data_type=data/image \
    --image_range 0 255 \
    --override_config '{"action_selection": {"n_actions": 5}}'