from typing import List, Union

import keras
import numpy as np
from keras import ops
from tqdm import tqdm

import ulsa.ops
import zea
from zea import Pipeline as ZeaPipeline
from zea.data.file import File
from zea.scan import Scan


class Pipeline(ZeaPipeline):
    def run(
        self,
        data: Union[File, np.ndarray, List[np.ndarray]],
        scan: Scan = None,
        keep_keys: list = None,
        verbose: bool = True,
        to_numpy: bool = True,
        selected_transmits=None,
        **params,
    ):
        # If Scan is provided, prepare parameters
        if scan is not None:
            # If selected_transmits provided, set them in the Scan
            if selected_transmits is not None:
                scan.set_transmits(selected_transmits)
            # If not provided, but scan has selected transmits, use those
            elif scan._selected_transmits is not None:
                selected_transmits = scan._selected_transmits

            params = self.prepare_parameters(scan=scan, **params)

        # If no keep_keys provided, default to ["maxval"]
        if keep_keys is None:
            keep_keys = ["maxval"]

        # If no selected_transmits provided, and could not be inferred from Scan, use all
        if selected_transmits is None:
            selected_transmits = slice(None)

        # If data is a File, load data
        if isinstance(data, File):
            data = data.load_data("raw_data", (slice(None), selected_transmits))

        if verbose:
            iterator = tqdm(data)
        else:
            iterator = data

        data_output = []
        for frame in iterator:
            output = self(data=frame, **params)
            processed_frame = output["data"]
            if to_numpy:
                processed_frame = ops.convert_to_numpy(processed_frame)
            data_output.append(processed_frame)
            for key in keep_keys:
                if key in output:
                    params[key] = output[key]
        if to_numpy:
            data_array = np.stack(data_output, axis=0)
        else:
            data_array = ops.stack(data_output, axis=0)
        return data_array, output

    def to_video_file(
        self,
        save_path: str,
        data: Union[File, np.ndarray, List[np.ndarray]],
        scan: Scan = None,
        keep_keys=None,
        verbose=True,
        dynamic_range=(-60, 0),
        frames_per_second=None,
        **kwargs,
    ):
        data_array, _ = self.run(
            data,
            scan=scan,
            keep_keys=keep_keys,
            verbose=verbose,
            to_numpy=True,
            dynamic_range=dynamic_range,
            **kwargs,
        )
        # Could be more RAM memory efficient to convert to 8bit frame by frame
        data_array = zea.display.to_8bit(data_array, dynamic_range, pillow=False)
        if frames_per_second is None and scan is not None:
            frames_per_second = scan.frames_per_second
        else:
            frames_per_second = 20  # default
        zea.io_lib.save_video(data_array, save_path, fps=frames_per_second)


def beamforming(rx_apo=True, pfield=False, low_pct=18, high_pct=95) -> list:
    """Create a pipeline for beamforming operations."""
    return [
        zea.ops.FirFilter(axis=-3, filter_key="bandpass_rf"),
        ulsa.ops.WaveletDenoise(),  # optional
        zea.ops.Demodulate(),
        zea.ops.LowPassFilter(complex_channels=True, axis=-2),  # optional
        zea.ops.Downsample(2),  # optional
        zea.ops.Map(
            [
                zea.ops.TOFCorrection(),
                zea.ops.PfieldWeighting() if pfield else zea.ops.Identity(),
                ulsa.ops.Multiply("rx_apo") if rx_apo else zea.ops.Identity(),
                zea.ops.DelayAndSum(),
            ],
            chunks=10,
            argnames=("flatgrid", "flat_pfield", "rx_apo"),
            in_axes=(0, 0, 1),
        ),
        zea.ops.ReshapeGrid(),
        zea.ops.EnvelopeDetect(),
        zea.ops.Normalize(),
        ulsa.ops.GetAutoDynamicRange(low_pct=low_pct, high_pct=high_pct),
        zea.ops.LogCompress(clip=False),
    ]


def resize(action_selection_shape: tuple, input_shape: tuple) -> list:
    """Resize to the shape that the prior model expects."""
    if action_selection_shape is None:
        return []
    ops = [zea.ops.keras_ops.Resize(size=action_selection_shape[:2], antialias=True)]
    if input_shape[:2] != action_selection_shape[:2]:
        pad = zea.ops.Pad(
            input_shape[:2], axis=(-3, -2), pad_kwargs=dict(mode="symmetric")
        )
        ops.append(pad)
    return ops


def make_pipeline(
    data_type,
    output_range=None,
    output_shape=None,
    action_selection_shape=None,
    jit_options="ops",
    with_batch_dim=False,
    rx_apo=True,
    low_pct=18,
    high_pct=95,
    **kwargs,
) -> Pipeline:
    if data_type == "data/raw_data":
        pipeline = Pipeline(
            [
                *beamforming(rx_apo=rx_apo, low_pct=low_pct, high_pct=high_pct),
                zea.ops.keras_ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(output_range),
                *resize(action_selection_shape, output_shape),
                zea.ops.keras_ops.Clip(x_min=output_range[0], x_max=output_range[1])
                if output_range is not None
                else zea.ops.Identity(),
            ],
            with_batch_dim=with_batch_dim,
            jit_options=jit_options,
            **kwargs,
        )
    elif data_type == "data/image":
        pipeline = Pipeline(
            [
                zea.ops.keras_ops.ExpandDims(axis=-1),
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
                zea.ops.keras_ops.ExpandDims(axis=-1),
                ulsa.ops.TranslateDynamicRange(output_range),
                # transpose so that azimuth dimension is on the outside, like a batch.
                # then we simply apply the 2d DM along all azimuthal angles
                zea.ops.keras_ops.Transpose(axes=(1, 0, 2, 3)),
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
