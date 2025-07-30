import keras

import ulsa.ops
import zea
from zea import Pipeline


def beamforming() -> list:
    """Create a pipeline for beamforming operations."""
    return [
        ulsa.ops.FirFilter(axis=-3, filter_key="bandpass_rf"),
        ulsa.ops.WaveletDenoise(),  # optional
        zea.ops.Demodulate(),
        ulsa.ops.LowPassFilter(complex_channels=True, axis=-2),  # optional
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
        ulsa.ops.LogCompressNoClip(),
    ]


def resize(action_selection_shape: tuple, input_shape: tuple) -> list:
    """Resize to the shape that the prior model expects."""
    ops = [ulsa.ops.Resize(size=action_selection_shape[:2])]
    if input_shape[:2] != action_selection_shape[:2]:
        pad = zea.ops.Pad(
            input_shape[:2], axis=(-3, -2), pad_kwargs=dict(mode="symmetric")
        )
        ops.append(pad)
    return ops


def make_pipeline(
    data_type,
    input_range,
    input_shape,
    action_selection_shape,
    jit_options="ops",
    with_batch_dim=False,
) -> Pipeline:
    if data_type not in ["data/image", "data/image_3D"]:
        pipeline = zea.Pipeline(
            [
                *beamforming(),
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(input_range),
                *resize(action_selection_shape, input_shape),
                zea.ops.Clip(*input_range),  # for resize and log compress clip
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
        )
    elif data_type == "data/image":
        pipeline = Pipeline(
            [
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(input_range),
                *resize(action_selection_shape, input_shape),
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
        )
    elif data_type == "data/image_3D":
        pipeline = Pipeline(
            [
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(input_range),
                # transpose so that azimuth dimension is on the outside, like a batch.
                # then we simply apply the 2d DM along all azimuthal angles
                zea.ops.Transpose((1, 0, 2, 3)),
                # we do cropping rather than resizing to maintain elevation focusing
                keras.layers.CenterCrop(*action_selection_shape),
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
        )

    return pipeline
