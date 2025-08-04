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
        ulsa.ops.GetAutoDynamicRange(),
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
    output_range,  # disable by setting to None
    output_shape,
    action_selection_shape,
    jit_options="ops",
    with_batch_dim=False,
    **kwargs,
) -> Pipeline:
    if data_type == "data/raw_data":
        pipeline = zea.Pipeline(
            [
                *beamforming(),
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(output_range),
                *resize(action_selection_shape, output_shape),
                zea.ops.Clip(
                    *(output_range if output_range else [])
                ),  # for resize and dynamic range clipping
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
            **kwargs,
        )
    elif data_type == "data/image":
        pipeline = Pipeline(
            [
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(output_range),
                *resize(action_selection_shape, output_shape),
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
            **kwargs,
        )
    elif data_type == "data/image_3D":
        pipeline = Pipeline(
            [
                ulsa.ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(output_range),
                # transpose so that azimuth dimension is on the outside, like a batch.
                # then we simply apply the 2d DM along all azimuthal angles
                zea.ops.Transpose((1, 0, 2, 3)),
                # we do cropping rather than resizing to maintain elevation focusing
                zea.ops.Lambda(
                    keras.layers.CenterCrop(*action_selection_shape),
                ),
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"Data type {data_type} not implemented in make_pipeline."
        )

    return pipeline
