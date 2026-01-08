# This file generates all the results from the in-house datasets.

# Fundamental
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir /mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_fundamental/ \
    --files /mnt/z/usbmd/Wessel/Verasonics/2024_USBMD_cardiac_S51_V2/P1/20240701_P1_A4CH_0001.hdf5 \
             /mnt/z/usbmd/Wessel/Verasonics/2024_USBMD_cardiac_S51_V2/P7/20240710_P7_A4CH_0000.hdf5 \
             /mnt/z/usbmd/Wessel/Verasonics/2024_USBMD_cardiac_S51_V2/P9/20241021_P9_A4CH_0000.hdf5 \
    --agent_config_path ./configs/cardiac_112_3_frames.yaml \
    --low_pct 18 \
    --high_pct 95

# Harmonic
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir /mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_harmonic/ \
    --files "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s1_a4ch_line_dw_0000.hdf5", \
            "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s2_a4ch_line_dw_0000.hdf5",\
            "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s3_a4ch_line_dw_0000.hdf5", \
    --agent_config_path ./configs/cardiac_112_3_frames_harmonic.yaml \
    --low_pct 44 \
    --high_pct 99.99

# Phantom
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir "/mnt/z/usbmd/Wessel/ulsa/eval_in_house/phantom/" \
    --files "/mnt/z/usbmd/Wessel/Verasonics/2025-11-18_zea/" \
    --low_pct 18 \
    --high_pct 95