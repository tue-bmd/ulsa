import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
import zea

zea.init_device()
sys.path.append("/ulsa")

import numpy as np
import scipy.signal
from keras import ops
from tqdm import tqdm

from active_sampling_temporal import fix_paths, preload_data
from ulsa.ops import FirFilter, LowPassFilter, WaveletDenoise
from zea.display import scan_convert_2d
from zea.utils import save_to_gif, translate

agent_config = zea.Config.from_yaml("configs/cardiac_112_3_frames.yaml")
agent_config = fix_paths(agent_config)
target_sequence = agent_config.data.target_sequence
data_type = agent_config.data.data_type
dynamic_range = agent_config.data.image_range

pipeline = zea.Pipeline(
    [
        FirFilter(axis=-3, filter_key="bandpass_rf"),
        WaveletDenoise(),  # optional
        zea.ops.Demodulate(),
        LowPassFilter(complex_channels=True, axis=-2),  # optional
        zea.ops.Downsample(2),  # optional
        zea.ops.PatchedGrid(
            [
                zea.ops.TOFCorrection(),
                # zea.ops.PfieldWeighting(),  # optional
                zea.ops.DelayAndSum(),
            ]
        ),
        zea.ops.EnvelopeDetect(),
        zea.ops.Normalize(),
        zea.ops.LogCompress(),
        zea.ops.Lambda(lambda x: ops.expand_dims(x, axis=-1)),
        zea.ops.Lambda(
            ops.image.resize,
            {
                "size": (112, 90),
                "interpolation": "bilinear",
                "antialias": True,
            },
        ),
        zea.ops.Lambda(lambda x: ops.squeeze(x, axis=-1)),
    ],
    with_batch_dim=False,
    jit_options="ops",
)


with zea.File(target_sequence) as file:
    n_frames = agent_config.io_config.get("frame_cutoff", "all")
    raw_data_sequence, scan, probe = preload_data(
        file,
        n_frames,
        data_type,
        type="diverging",
    )

images = []
bandpass_rf = scipy.signal.firwin(
    numtaps=128,
    cutoff=np.array([0.5, 1.5]) * scan.center_frequency,
    pass_zero="bandpass",
    fs=scan.sampling_frequency,
)
scan.polar_limits = list(np.deg2rad([-45, 45]))
scan.n_x = 90
scan.dynamic_range = [-70, -30]
scan.fill_value = float(scan.dynamic_range[0])
params = pipeline.prepare_parameters(
    scan=scan,
    bandwidth=2e6,
    bandpass_rf=bandpass_rf,
)
for raw_data in tqdm(raw_data_sequence):
    output = pipeline(data=raw_data, **params)
    image = output.pop("data")
    params["maxval"] = output.pop("maxval")
    image, _ = scan_convert_2d(
        image,
        rho_range=(0, image.shape[0]),
        theta_range=scan.theta_range,
        resolution=0.1,
        fill_value=np.nan,
        order=0,
    )
    image = translate(image, scan.dynamic_range, (0, 255))
    image = ops.clip(image, 0, 255)
    image = ops.cast(image, dtype="uint8")
    images.append(ops.convert_to_numpy(image))

save_to_gif(images, "diverging.gif", fps=agent_config.io_config.gif_fps)
