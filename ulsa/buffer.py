"""Managing a frame buffer which can be used in jax."""

from jax import tree_util
from keras import ops


@tree_util.register_pytree_node_class
class FrameBuffer:
    def __init__(self, image_shape, batch_size=None, buffer_size=2):
        """
        image_shape: [h, w, c]
        buffer_size: number of temporal frames stored
        """
        self.image_shape = tuple(image_shape)
        self.buffer_size = buffer_size
        sample_shape = (
            image_shape[:-1] if batch_size is None else [batch_size, *image_shape[:-1]]
        )
        self.buffer = ops.zeros(
            [*sample_shape, buffer_size]
        )  # shape: [h, w, buffer_size]

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer = self.buffer.at[key].set(value)
        return self.buffer[key]

    def shift(self, new_item):
        self.buffer = fifo_shift(self.buffer, new_item)

    def latest(self):
        return self.buffer[..., -1]

    def is_populated(self):
        return not ops.all(self.buffer[..., -2] == 0) and not ops.all(
            self.buffer[..., -1] == 0
        )

    # 👇 Register buffer as child, and image_shape/buffer_size as aux
    def tree_flatten(self):
        children = (self.buffer,)
        aux_data = (self.image_shape, self.buffer_size)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        image_shape, buffer_size = aux_data
        obj = cls.__new__(cls)
        obj.image_shape = image_shape
        obj.buffer_size = buffer_size
        obj.buffer = children[0]
        return obj


def fifo_shift(existing_buffer, new_item):
    return ops.concatenate([existing_buffer[..., 1:], new_item], axis=-1)
