from typing import List

from jax import tree_util
from keras import ops


@tree_util.register_pytree_node_class
class FrameBuffer:
    def __init__(
        self,
        image_shape: List[int] | int,
        buffer_size: int,
        batch_size: int | None = None,
        axis: int = -1,
    ):
        self.buffer_size = buffer_size
        if not isinstance(image_shape, (list, tuple)):
            image_shape = [image_shape]
        sample_shape = image_shape if batch_size is None else [batch_size, *image_shape]
        self.buffer = ops.zeros([*sample_shape, buffer_size])
        self.axis = axis
        self.buffer = ops.moveaxis(self.buffer, -1, axis)

    @property
    def shape(self):
        return self.buffer.shape

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer = self.buffer.at[key].set(value)
        return self.buffer[key]

    def shift(self, new_item):
        self.buffer = fifo_shift(self.buffer, new_item, self.axis)

    def latest(self):
        return ops.take(self.buffer, -1, axis=self.axis)

    # 👇 Register buffer as child, and image_shape/buffer_size as aux
    def tree_flatten(self):
        children = (self.buffer,)
        aux_data = (self.buffer_size, self.axis)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # unpack
        (buffer,) = children
        (buffer_size, axis) = aux_data

        # create new instance and set attributes
        obj = cls.__new__(cls)
        obj.buffer_size = buffer_size
        obj.axis = axis
        obj.buffer = buffer
        return obj


def fifo_shift(existing_buffer, new_item, axis=-1):
    # Add buffer dimension if possible
    if new_item.shape[axis] != 1:
        if ops.ndim(new_item) == ops.ndim(existing_buffer) - 1:
            new_item = ops.expand_dims(new_item, axis=axis)
        else:
            raise ValueError(
                f"new_item must have shape 1 in axis={axis}. Got shape "
                + str(new_item.shape)
            )

    # Support buffer_size == 1 (no shift, just replace)
    if existing_buffer.shape[axis] == 1:
        return new_item

    shifted = ops.take(
        existing_buffer, indices=range(1, existing_buffer.shape[axis]), axis=axis
    )
    return ops.concatenate([shifted, new_item], axis=axis)
