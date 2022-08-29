import json
import os
import sys

"""
python3 data/alfred_utils/filter_metadata.py \
    -dd /data/anthony/alfred/data/json_2.1.0/ \
    --num_processes 20
"""

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "gen"))

from env.thor_env import ThorEnv
import glob
import tqdm
import random
import multiprocessing
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
from replay_json import split_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-dd", type=str, help="where is traj json files located"
    )
    parser.add_argument(
        "--num_processes", type=int, help="number of parallel processes"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    # don't use more cpus than you have
    if args.debug:
        num_processes = 1
    else:
        num_processes = np.minimum(args.num_processes, multiprocessing.cpu_count())

    all_json_files = list(glob.glob(os.path.join(args.data_dir, "*/*/*/*.json")))

    random.shuffle(all_json_files)

    print(f"{len(all_json_files)} files found")

    def filter_metadata(json_files):
        total = len(json_files)
        for i, json_file in enumerate(json_files):
            og_metadata_f = os.path.join(
                os.path.dirname(json_file), "traj_metadata.pkl"
            )
            small_metadata_f = os.path.join(
                os.path.dirname(json_file), "traj_metadata_small.pkl"
            )
            filter_metadata_f = os.path.join(
                os.path.dirname(json_file), "traj_metadata_filtered.pkl"
            )

            # filter only visible objects
            def filter_visible(list_of_objs):
                return [obj for obj in list_of_objs if obj["visible"]]

            if (
                not os.path.exists(small_metadata_f)
                and not os.path.exists(filter_metadata_f)
                and os.path.exists(og_metadata_f)
            ):
                data = pickle.load(open(og_metadata_f, "rb"))
                save = [event.metadata for event in data]
                pickle.dump(save, open(small_metadata_f, "wb"))

                for d in save:
                    d["objects"] = filter_visible(d["objects"])
                pickle.dump(save, open(filter_metadata_f, "wb"))

                if os.path.exists(og_metadata_f):
                    os.remove(og_metadata_f)

            if os.path.exists(small_metadata_f) and not os.path.exists(
                filter_metadata_f
            ):
                data = pickle.load(open(small_metadata_f, "rb"))

                for d in data:
                    d["objects"] = filter_visible(d["objects"])

                pickle.dump(data, open(filter_metadata_f, "wb"))

                # delete the OG large metadata file
                if os.path.exists(og_metadata_f):
                    os.remove(og_metadata_f)

            if os.path.exists(small_metadata_f) and os.path.exists(filter_metadata_f):
                if os.path.exists(og_metadata_f):
                    os.remove(og_metadata_f)

            print(f"{i+1}/{total} completed")

    if args.debug:
        filter_metadata(all_json_files)
    else:
        file_chunks = list(split_list(all_json_files, num_processes))
        print(f"{len(file_chunks)} chunks")

        with multiprocessing.Pool(num_processes) as p:
            results = p.map(filter_metadata, file_chunks)
            p.close()
            p.join()
