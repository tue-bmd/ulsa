import numpy as np
import SimpleITK as sitk

import zea


def undo_log_compress(data):
    return 10 ** (data / 20)


def auto_dynamic_range(data, low_pct=44, high_pct=99.99):
    # NOTE data is log compressed data
    linear_data = undo_log_compress(data)
    low_val = np.nanquantile(linear_data, low_pct / 100)
    low_val = 20 * np.log10(low_val)
    high_val = np.nanquantile(linear_data, high_pct / 100)
    high_val = 20 * np.log10(high_val)
    return (low_val, high_val)


def npz_to_itk(npz_path, itk_path, dynamic_range="file", resolution=0.3):
    data = np.load(npz_path, allow_pickle=True)
    reconstuctions = data["reconstructions"]
    theta_range = data["theta_range"]

    if dynamic_range == "file":
        dynamic_range = data["dynamic_range"]
    elif dynamic_range == "auto":
        dynamic_range = auto_dynamic_range(reconstuctions, low_pct=44, high_pct=99.99)

    if dynamic_range is not None:
        reconstuctions = np.clip(reconstuctions, dynamic_range[0], dynamic_range[1])
        fill_value = dynamic_range[0]
    else:
        fill_value = reconstuctions.min()

    reconstuctions, _ = zea.display.scan_convert_2d(
        reconstuctions,
        (0, reconstuctions.shape[1]),
        theta_range,
        resolution=resolution,
        fill_value=fill_value,
        order=0,
    )
    if dynamic_range is not None:
        reconstuctions = zea.display.to_8bit(
            reconstuctions, dynamic_range, pillow=False
        )

    sitk.WriteImage(
        sitk.GetImageFromArray(reconstuctions),
        itk_path,  # .nii.gz extension for compression
    )


if __name__ == "__main__":
    npz_to_itk(
        "/mnt/z/usbmd/Wessel/ulsa/eval_in_house_cardiac_v3/20251222_s2_a4ch_line_dw_0000/diverging.npz",
        "/mnt/z/usbmd/Wessel/ulsa/eval_in_house_cardiac_v3/20251222_s2_a4ch_line_dw_0000/diverging.nii.gz",
        dynamic_range=None,
    )
