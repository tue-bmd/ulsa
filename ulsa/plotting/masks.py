import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from zea.tools.selection_tool import remove_masks_from_axs
from zea.visualize import plot_shape_from_mask


def update_imshow_with_masks(
    frame_no: int,
    axs: matplotlib.axes.Axes,
    imshow_obj: matplotlib.image.AxesImage,
    images: np.ndarray,
    masks: np.ndarray,
):
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta"]
    imshow_obj.set_array(images[frame_no])

    remove_masks_from_axs(axs)

    for _masks, color in zip(masks, colors):
        plot_shape_from_mask(
            axs,
            _masks[frame_no],
            alpha=0.5,
            facecolor=color,
            edgecolor=color,
            linewidth=2.0,
        )


def visualize_masks(images, valve, myocardium, ventricle, filepath, fps=10):
    def update(frame_no):
        update_imshow_with_masks(
            frame_no,
            axs,
            imshow_obj,
            images,
            [
                ventricle,
                myocardium,
                valve,
            ],
        )

    fig, axs = plt.subplots()
    imshow_obj = axs.imshow(images[0], cmap="gray")
    interval = 1000 / fps  # milliseconds
    ani = FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=interval,
    )
    ani.save(filepath, writer="pillow")
