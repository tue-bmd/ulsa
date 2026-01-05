import os

os.environ["KERAS_BACKEND"] = "numpy"
import sys

import numpy as np
import SimpleITK as sitk

sys.path.append("/ulsa")
from ulsa.metrics import gcnr_per_frame

mask = sitk.ReadImage(
    "/mnt/z/usbmd/Wessel/eval_in_house_cardiac_v3/20251222_s3_a4ch_line_dw_0000/focused_annotated.nii.gz"
)
mask = sitk.GetArrayFromImage(mask)

background = mask == 0
ventricle = mask == 1
myocardium = mask == 2
valve = mask == 3
other = mask > 3
assert np.all(other == False), "Unexpected label in mask"
