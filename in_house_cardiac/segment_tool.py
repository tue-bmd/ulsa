if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "numpy"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from zea.internal.viewer import (
    filename_from_window_dialog,
    get_matplotlib_figure_props,
    move_matplotlib_figure,
)
from zea.io_lib import _SUPPORTED_VID_TYPES, load_image, load_video
from zea.tools.selection_tool import (
    add_rectangle_from_mask,
    add_shape_from_mask,
    ask_for_num_selections,
    ask_for_selection_tool,
    ask_save_animation_with_fps,
    interactive_selector,
    interactive_selector_with_plot_and_metric,
    interpolate_masks,
    update_imshow_with_mask,
)


def ask_for_title():
    print("What are you selecting?")
    title = input("Enter a title for the selection: ")
    if not title:
        raise ValueError("Title cannot be empty.")
    # Convert title to snake_case
    title = title.strip().replace(" ", "_").lower()
    print(f"Title set to: {title}")
    return title


def main():
    """Main function for interactive selector on multiple images."""
    print(
        "Select as many images as you like, OR select 1 video / gif, "
        "and close window to continue..."
    )
    images = []
    file_names = []
    try:
        while True:
            file = filename_from_window_dialog("Choose image / video file")
            if file.suffix in [".png", ".jpg", ".jpeg"]:
                image = load_image(file)
                images.append(image)
                file_names.append(file.name)
                same_images = True
            elif file.suffix in _SUPPORTED_VID_TYPES:
                images.extend(load_video(file))
                same_images = False
                break
    except Exception as e:
        if len(images) == 0:
            raise e
        print("No more images selected. Continuing...")

    title = ask_for_title()
    selector = ask_for_selection_tool()

    if same_images is True:
        figs, axs = [], []
        for i, (image, file_name) in enumerate(zip(images[::-1], file_names[::-1])):
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            if i == len(images) - 1:
                ax.set_title(f"Make selection in this plot\n {file_name}")
            else:
                ax.set_title(file_name)
            ax.axis("off")
            axs.append(ax)
            figs.append(fig)

        axs = axs[::-1]
        figs = figs[::-1]

        interactive_selector_with_plot_and_metric(
            images,
            axs,
            selector=selector,
            metric="gcnr",
        )

    else:
        if len(images) > 3:
            print(f"Found sequence of {len(images)} images. ")

            num_selections = ask_for_num_selections()

            selection_idx = np.linspace(0, len(images) - 1, int(num_selections)).astype(
                int
            )
            selection_images = [images[idx] for idx in selection_idx]
            selection_masks = []
            pos, size = None, None
            for image in selection_images:
                fig, axs = plt.subplots()
                fig.tight_layout()
                # set window size to what user selected for plot before
                if pos is not None:
                    move_matplotlib_figure(fig, pos, size)

                axs.imshow(image, cmap="gray")

                while True:
                    _, mask = interactive_selector(
                        image, axs, selector=selector, num_selections=1
                    )
                    # check if mask is empty else retry
                    if mask[0].sum() == 0:
                        print(
                            "Empty mask. Try again, make sure to make a descent selection..."
                        )
                    else:
                        break

                pos, size = get_matplotlib_figure_props(fig)

                if selector == "rectangle":
                    add_rectangle_from_mask(axs, mask[0], alpha=0.5)
                else:
                    add_shape_from_mask(axs, mask[0], alpha=0.5)
                plt.close()
                selection_masks.append(mask[0])

        # small hack to make sure that there is always at least two masks for interpolation
        if len(selection_masks) == 1:
            selection_masks.append(selection_masks[0])

        interpolated_masks = interpolate_masks(
            selection_masks, num_frames=len(images), rectangle=(selector == "rectangle")
        )

        fig, axs = plt.subplots()

        imshow_obj = axs.imshow(images[0], cmap="gray")

        if selector == "rectangle":
            add_rectangle_from_mask(axs, interpolated_masks[0])
        else:
            add_shape_from_mask(axs, interpolated_masks[0], alpha=0.5)

        filestem = Path(file.parent.stem + "_" + f"{file.stem}_{title}_annotations.gif")
        np.save(filestem.with_suffix(".npy"), interpolated_masks)
        print(f"Succesfully saved interpolated masks to {filestem.with_suffix('.npy')}")

        fps = ask_save_animation_with_fps()

        ani = FuncAnimation(
            fig,
            update_imshow_with_mask,
            frames=len(images),
            fargs=(axs, imshow_obj, images, interpolated_masks, selector),
            interval=1000 / fps,
        )
        filename = filestem.with_suffix(".gif")
        ani.save(filename, writer="pillow")
        print(f"Succesfully saved animation as {filename}")


if __name__ == "__main__":
    main()
