from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea.internal.viewer import filename_from_window_dialog
from zea.io_lib import _SUPPORTED_VID_TYPES, load_video


def select_frames(arr):
    """
    Show each frame from a numpy array of shape (frames, h, w).
    Press 'y' to accept a frame. Returns a list of selected frame indices.
    """
    selected = []
    num_frames = len(arr)

    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(arr[i], cmap="gray")
        ax.set_title(f'Frame {i} - Press "y" to accept, any other key to skip')
        accepted = []

        def on_key(event):
            if event.key == "y":
                accepted.append(True)
            plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if accepted:
            selected.append(i)

    return selected


def ask_for_video():
    print("Select a video or gif file to load frames from.")
    file = filename_from_window_dialog("Choose video / gif file")
    if not file or file.suffix not in _SUPPORTED_VID_TYPES:
        raise ValueError("Only video or gif files are accepted.")
    images = load_video(file)
    return images, file


def ask_for_save_path():
    print("Select a folder to save the selected frames.")
    save_path = filename_from_window_dialog(
        "Choose folder to save selected frames", mode="directory"
    )
    if not save_path:
        raise ValueError("No save path selected.")
    return Path(save_path)


def main():
    images, filepath = ask_for_video()
    save_path = ask_for_save_path()
    filename = filepath.stem
    selected_idx = select_frames(images)
    np.save(save_path / f"{filename}_selected_frames.npy", selected_idx)
    zea.utils.save_to_mp4(
        images[selected_idx],
        save_path / f"{filename}_selected_frames.mp4",
    )


if __name__ == "__main__":
    main()
