# This file generates all the results from the in-house datasets.

# Fundamental
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir /mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_fundamental \
    --folder /mnt/z/usbmd/Wessel/Verasonics/2024_USBMD_cardiac_S51_V2/ \
    --pattern "*_A4CH_*.hdf5" \
    --low_pct 18 \
    --high_pct 95

# Harmonic
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir /mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_harmonic \
    --folder /mnt/datasets/2026_USBMD_A4CH_S51_V2/ \
    --pattern "*_a4ch_line_dw_*.hdf5" \
    --low_pct 44 \
    --high_pct 99.99

# Phantom
python benchmarking_scripts/eval_in_house_cardiac.py \
    --save_dir "/mnt/z/usbmd/Wessel/ulsa/eval_in_house/phantom" \
    --folder "/mnt/z/usbmd/Wessel/Verasonics/2025-11-18_zea" \
    --pattern "*.hdf5" \
    --low_pct 18 \
    --high_pct 95