import json
import os
import sys

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "gen"))
from env.thor_env import ThorEnv
import glob
import pickle
import tqdm
import multiprocessing
import numpy as np


def replay_json(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # setup
    scene_num = traj_data["scene"]["scene_num"]
    object_poses = traj_data["scene"]["object_poses"]
    dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
    object_toggles = traj_data["scene"]["object_toggles"]

    scene_name = "FloorPlan%d" % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # store the events
    events = []

    # initialize
    event = env.step(dict(traj_data["scene"]["init_action"]))
    events.append(event)
    # import ipdb; ipdb.set_trace()
    # print("Task: %s" % (traj_data['turk_annotations']['anns'][0]['task_desc']))

    steps_taken = 0
    for ll_action in traj_data["plan"]["low_actions"]:
        hl_action_idx, traj_api_cmd, traj_discrete_action = (
            ll_action["high_idx"],
            ll_action["api_action"],
            ll_action["discrete_action"],
        )

        # print templated low-level instructions & discrete action
        # print("HL Templ: %s, LL Cmd: %s" % (traj_data['turk_annotations']['anns'][0]['high_descs'][hl_action_idx],
        #                                     traj_discrete_action['action']))

        # Use the va_interact that modelers will have to use at inference time.
        action_name, action_args = (
            traj_discrete_action["action"],
            traj_discrete_action["args"],
        )

        # three ways to specify object of interest mask
        # 1. create a rectangular mask from bbox
        # mask = env.bbox_to_mask(action_args['bbox']) if 'bbox' in action_args else None  # some commands don't require any arguments
        # 2. create a point mask from bbox
        # mask = env.point_to_mask(action_args['point']) if 'point' in action_args else None
        # 3. use full pixel-wise segmentation mask
        compressed_mask = action_args["mask"] if "mask" in action_args else None
        if compressed_mask is not None:
            mask = env.decompress_mask(compressed_mask)
        else:
            mask = None

        success, event, target_instance_id, err, _ = env.va_interact(
            action_name, interact_mask=mask
        )

        events.append(event)
        if not success:
            raise RuntimeError(err)

        steps_taken += 1

    return steps_taken, events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-dd", type=str, help="where is traj json files located"
    )
    args = parser.parse_args()

    cpus = 20

    # don't use more cpus than you have
    cpus = np.minimum(cpus, multiprocessing.cpu_count())

    all_json_files = list(glob.glob(os.path.join(args.data_dir, "*/*/*/*.json")))

    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [
            alist[i * length // wanted_parts : (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)
        ]

    file_chunks = list(split_list(all_json_files, cpus))

    def generate_traj_metadata(json_files):
        env = ThorEnv()

        total = len(json_files)
        for i, json_file in enumerate(json_files):

            parts = json_file.replace(args.data_dir, "").split("/")

            phase, task_type, trial_id, f_name = parts

            # save
            metadata_f = os.path.join(os.path.dirname(json_file), "traj_metadata.pkl")

            if not os.path.exists(metadata_f):
                # replay json file and save metadata
                steps, event_metadata = replay_json(env, json_file)
                print(steps, len(event_metadata))

                pickle.dump(event_metadata, open(metadata_f, "wb"))

            print(f"{i+1}/{total} completed")

    with multiprocessing.Pool(cpus) as p:
        results = p.map(generate_traj_metadata, file_chunks)
        p.close()
        p.join()
