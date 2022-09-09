import os
import cv2
import numpy as np


def save_video_sequence(file, frames, fps=10.0):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    size = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(
        file, cv2.VideoWriter_fourcc(*"XVID"), fps, size, isColor=True
    )
    for img in frames:
        video.write(img.astype(np.uint8))
    video.release()


def save_episode_as_video(episode, filename, caption="", fps=1.0):
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in range(len(episode["image"])):
        img = episode["image"][t]

        # resize image
        img = cv2.resize(
            ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8),
            (500, 500),
        )

        # add timestep to bottom right
        cv2.putText(
            img,
            f"t = {t}",
            (400, 450),
            font,
            color=(0, 0, 0),
            fontScale=0.5,
            thickness=1,
        )

        text = ""

        keys = [
            "curr_instr",
            "progress_pred",
            "binary_token",
            "action_str",
            "obj_str",
            "info.action_success",
            "info.err",
            "info.target_instance_id",
            "info.num_fails",
        ]

        for key in keys:
            keys = key.split(".")

            if len(keys) == 1:
                if key in episode and episode[key][t] is not None:
                    text += f"{key}: {episode[key][t]}\n"
            elif len(keys) == 2:
                if (
                    keys[0] in episode
                    and keys[1] in episode[keys[0]][t]
                    and episode[keys[0]][t][keys[1]] is not None
                ):
                    text += f"{key}: {episode[keys[0]][t][keys[1]]}\n"

        # add caption text
        y0, dy = 50, 20
        for j, line in enumerate(text.split("\n")):
            y = y0 + j * dy
            cv2.putText(
                img, line, (50, y), font, color=(0, 0, 0), fontScale=0.5, thickness=2
            )

        img = img[:, :, ::-1]
        frames.append(img)

    # pad end
    for _ in range(25):
        frames.append(img)
    save_video_sequence(filename, frames, fps=fps)
