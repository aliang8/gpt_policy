import json
import os
import sys
from data.alfred import ALFREDDataset
import data.alfred_utils.data_utils as utils

"""
python3 data/alfred_utils/gather_metadata.py \
    -dd /data/anthony/alfred/data/json_2.1.0/ 
"""

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "gen"))

from env.thor_env import ThorEnv
import re
import glob
import tqdm
import random
import lmdb
import time
import copy
import torch
import shutil
import threading
import numpy as np
from pathlib import Path
from progressbar import ProgressBar

try:
    import cPickle as pickle
except:
    import pickle
from pympler import asizeof
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

def gather_feats(files, output_path):
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_feats = lmdb.open(str(output_path), 100 * 35 * 31 * 2000, writemap=True)
    with lmdb_feats.begin(write=True) as txn_feats:
        for idx, path in tqdm(enumerate(files)):
            traj_feats = torch.load(path).numpy()
            txn_feats.put("{:06}".format(idx).encode("ascii"), traj_feats.tobytes())
    lmdb_feats.close()

def gather_jsons(files, output_path):
    if output_path.exists():
        os.remove(output_path)
    jsons = {}
    for idx, path in tqdm.tqdm(enumerate(files)):
        with open(path, "rb") as f:
            jsons_idx = pickle.load(f)
            jsons["{:06}".format(idx).encode("ascii")] = jsons_idx
    with output_path.open("wb") as f:
        pickle.dump(jsons, f)


def get_json_files(input_path, processed_files_path, fast_epoch):
    if (input_path / "processed.txt").exists():
        # the dataset was generated locally
        with (input_path / "processed.txt").open() as f:
            json_files = [line.strip() for line in f.readlines()]
            json_files = [
                line.split(";")[0] for line in json_files if line.split(";")[1] == "1"
            ]
            json_files = [str(input_path / line) for line in json_files]
    else:
        # the dataset was downloaded from ALFRED servers
        json_files_all = sorted([str(path) for path in input_path.glob("*/*/*/*.json")])
        print(len(json_files_all))
        json_files = json_files_all
    if fast_epoch:
        json_files = json_files[::20]
    num_files = len(json_files)
    # if processed_files_path is not None and processed_files_path.exists():
    #     with processed_files_path.open() as f:
    #         processed_files = set([line.strip() for line in f.readlines()])
    #     json_files = [traj for traj in json_files if traj not in processed_files]
    # json_files = [Path(path) for path in json_files]
    return json_files, num_files


def run_in_parallel(func, num_workers, output_path, args, use_processes=False):
    if num_workers == 0:
        args.append(output_path / "worker00")
        func(*args)
    else:
        threads = []
        for idx in range(num_workers):
            args_worker = copy.copy(args) + [output_path / "worker{:02d}".format(idx)]
            if not use_processes:
                ThreadClass = threading.Thread
            else:
                ThreadClass = torch.multiprocessing.Process
            thread = ThreadClass(target=func, args=args_worker)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()


def process_jsons(json_files, lock, save_path):
    save_path.mkdir(exist_ok=True)
    (save_path / "jsons").mkdir(exist_ok=True)
    if str(save_path).endswith("/worker00"):
        with lock:
            progressbar = ProgressBar(maxval=len(json_files))
            progressbar.start()
    while True:
        with lock:
            if len(json_files) == 0:
                break
            traj_path = Path(json_files.pop())
        with traj_path.open() as f:
            traj_orig = json.load(f)

        # save masks and traj jsons
        filename = "{}:{}".format(
            traj_path.parts[-2], re.sub(".json", ".pkl", traj_path.name)
        )
        with (save_path / "jsons" / filename).open("wb") as f:
            pickle.dump(traj_orig, f)

        # report the progress
        with lock:
            utils.update_log(
                save_path.parents[0], stage="jsons", update="increase", progress=1
            )
            if str(save_path).endswith("/worker00"):
                progressbar.update(progressbar.maxval - len(json_files))
    if str(save_path).endswith("/worker00"):
        progressbar.finish()

def process_feats(json_files, extractor, lock, image_folder, save_path):
    (save_path / "feats").mkdir(exist_ok=True)
    if str(save_path).endswith("/worker00"):
        with lock:
            progressbar = ProgressBar(max_value=json_files.qsize())
            progressbar.start()
    while True:
        with lock:
            if json_files.qsize() == 0:
                break
            traj_path = Path(json_files.get())
            
        filename_new = "{}:{}".format(traj_path.parts[-2], re.sub(".json", ".npy", traj_path.name))
        
        state_feat = 
        
        
        
        if feat is not None:
            torch.save(feat, save_path / "feats" / filename_new)
        with lock:
            with open(save_path.parents[0] / "processed_feats.txt", "a") as f:
                f.write(str(traj_path) + "\n")
            model_util.update_log(save_path.parents[0], stage="feats", update="increase", progress=1)
            if str(save_path).endswith("/worker00"):
                progressbar.update(progressbar.max_value - json_files.qsize())
    if str(save_path).endswith("/worker00"):
        progressbar.finish()

def gather_data(output_path, num_workers):
    # create output dir for jsons
    for dirname in ["jsons"]:
        if (output_path / dirname).is_dir():
            shutil.rmtree(output_path / dirname)
        (output_path / dirname).mkdir()

    for dirname in ["jsons"]:
        for path_file in output_path.glob("worker*/{}/*".format(dirname)):
            if path_file.stat().st_size == 0:
                continue
            path_symlink = output_path / dirname / path_file.name
            link_file = True
            if path_symlink.is_symlink():
                # this file was already linked
                if path_file.stat().st_size > path_symlink.stat().st_size:
                    # we should replace the previously linked file with a new one
                    link_file = True
                    path_symlink.unlink()
                else:
                    # we should keep the previously linked file
                    link_file = False
            if link_file:
                path_symlink.symlink_to(path_file)

    partitions = ("train", "valid_seen", "valid_unseen", "test_seen", "test_unseen")
    if not (output_path / ".deleting_worker_dirs").exists():
        for partition in partitions:
            print("Processing %s trajectories" % partition)
            feats_files = output_path.glob("feats/{}:*.pt".format(partition))
            feats_files = sorted([str(path) for path in feats_files])
            jsons_files = [
                p.replace("/feats/", "/jsons/").replace(".pt", ".pkl")
                for p in feats_files
            ]
            (output_path / partition).mkdir(exist_ok=True)
            gather_jsons(jsons_files, output_path / partition / "jsons.pkl")

    print("Removing worker directories")
    (output_path / ".deleting_worker_dirs").touch()
    for worker_idx in range(max(num_workers, 1)):
        worker_dir = output_path / "worker{:02d}".format(worker_idx)
        shutil.rmtree(worker_dir)
    for dirname in ["jsons"]:
        shutil.rmtree(output_path / dirname)
    os.remove(output_path / ".deleting_worker_dirs")
    # os.remove(output_path / "processed_feats.txt")


def main():
    input_path = Path(args.data_dir)
    output_path = Path(args.save_dir)

    output_path.mkdir(exist_ok=True)
    print("Creating a dataset {} using data from {}".format(output_path, input_path))

    json_files, num_files = get_json_files(
        input_path, output_path / "processed.txt", True
    )

    utils.save_log(
        output_path,
        progress=num_files - len(json_files),
        total=num_files,
        stage="jsons",
    )

    print(
        "Creating a dataset with {} trajectories using {} workers".format(
            num_files, args.num_workers
        )
    )
    print(
        "Processing JSONs and masks ({} were already processed)".format(
            num_files - len(json_files)
        )
    )

    if len(json_files) > 0:
        lock = threading.Lock()
        run_in_parallel(
            process_jsons,
            args.num_workers,
            output_path,
            args=[json_files, lock],
        )

    # finally, gather all the data
    gather_data(output_path, args.num_workers)
    print("The dataset was saved to {}".format(output_path))

    # read which features need to be extracted
    trajs_list, num_files_again = get_json_files(
        input_path, output_path / "processed_feats.txt", True
    )

    utils.save_log(
        output_path,
        progress=num_files - len(trajs_list),
        total=num_files,
        stage="feats",
    )
    print(
        "Extracting features ({} were already processed)".format(
            num_files - len(trajs_list)
        )
    )

    if len(trajs_list) > 0:
        manager = torch.multiprocessing.Manager()
        lock = manager.Lock()
        trajs_queue = manager.Queue()
        for path in trajs_list:
            trajs_queue.put(path)
        args_process_feats = [trajs_queue, extractor, lock, args.image_folder]
        run_in_parallel(
            process_feats,
            args.num_workers,
            output_path,
            args=args_process_feats,
            use_processes=True,
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-dd", type=str, help="where is traj json files located"
    )
    parser.add_argument(
        "--save_dir", "-sd", type=str, help="where processed features should be located"
    )
    parser.add_argument(
        "--num_workers", "-nw", type=int, default=4, help="number of workers"
    )

    args = parser.parse_args()

    # hparams = AttrDict(
    #     data_dir=args.data_dir,
    #     decoder_model_cls="gpt2",
    #     load_lang=True,
    #     chunk_size=512,
    #     alfred_split="valid_seen",
    #     load_frac_done=False,
    # )
    # dataset = ALFREDDataset(hparams=hparams)

    # # figure out the max visible objs
    # max_visible_objects = 0
    # map_size = 100 * 35 * 31 * 2000

    # # create a new LMDB DB
    # db = lmdb.open(
    #     os.path.join(args.data_dir, "alfred_metadata_lmdb"), map_size=map_size
    # )

    # for data in dataset.data:
    #     # state encoding
    #     num_steps = len(data["metadata"])
    #     visible_obj_metadatas = [
    #         data["metadata"][t]["objects"] for t in range(num_steps)
    #     ]

    #     # encode only visible objects
    #     max_visible_objects_ = max([len(objs) for objs in visible_obj_metadatas])
    #     max_visible_objects = max(max_visible_objects, max_visible_objects_)

    # print(f"max visible objects: {max_visible_objects}")

    # trial_ids = []
    # with db.begin(write=True) as txn:
    #     for data in tqdm.tqdm(dataset.data):
    #         num_steps = len(data["metadata"])
    #         visible_obj_metadatas = [
    #             data["metadata"][t]["objects"] for t in range(num_steps)
    #         ]

    #         state_encodings = []
    #         for t in range(num_steps):
    #             if visible_obj_metadatas[t]:
    #                 state_encoding = dataset.encode_list_objects(
    #                     visible_obj_metadatas[t], max_visible_objects
    #                 )
    #                 state_encodings.append(state_encoding)
    #             else:
    #                 state_encodings.append(np.zeros((max_visible_objects, 31)))

    #         state_encodings = np.array(state_encodings[1:])  # skip the first step
    #         key = data["task_id"]
    #         trial_ids.append(key)
    #         txn.put(key.encode("ascii"), state_encodings.tobytes())

    # db.close()

    # # read lmdb
    # start = time.time()
    # db = lmdb.open(
    #     os.path.join(args.data_dir, "alfred_metadata_lmdb"),
    #     readonly=True,
    #     lock=False,
    #     readahead=False,
    #     meminit=False,
    #     max_readers=4,
    # )

    # print(trial_ids[:10])
    # with db.begin() as txn:
    #     for i in range(len(dataset.data)):
    #         data = txn.get(trial_ids[i].encode("ascii"))
    #         metadata = np.frombuffer(data)
    # db.close()

    # print(time.time() - start)

    # first process jsons

    # read which files need to be processed
