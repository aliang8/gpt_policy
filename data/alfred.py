import os
import glob
import json
import random
from enum import Enum
import numpy as np
from data.dataset import BaseDataset, SingleSequenceDataset
from data.alfred_utils.object_encoder import encode_object
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from utils.lang_utils import get_tokenizer
from typing import Optional, Dict, List, Any

# from ai2thor.interact import DefaultActions
import sys

sys.path.append("/home/anthony/alfred")
import gen.constants as constants


class Actions(Enum):
    MoveRight = 0
    MoveLeft = 1
    MoveAhead = 2
    MoveBack = 3
    LookUp = 4
    LookDown = 5
    RotateRight = 8
    RotateLeft = 9
    PutObject = 10
    PickupObject = 11
    OpenObject = 12
    CloseObject = 13
    ToggleObjectOff = 14
    ToggleObjectOn = 15
    SliceObject = 16


class ALFREDSingleSequenceDataset(SingleSequenceDataset):
    def __init__(self, hparams: Dict, split="train", *args, **kwargs):
        self.hparams = hparams
        self.split = split

        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        self.data = self.load_data()

        super().__init__(*args, **kwargs)

    def load_data(self):
        data = []
        splits = os.listdir(self.hparams.alfred_data_dir)
        data_dir = os.path.join(self.hparams.alfred_data_dir, self.split)
        tasks = glob.glob(f"{data_dir}/*/*")

        print(f"{len(tasks)} trajectories found for {self.split}")

        for task in tasks:  # doesn't work for test
            trial_id = os.path.basename(task)
            task_name = os.path.basename(os.path.dirname(task))
            (
                task_type,
                obj,
                moveable_receptacle,
                receptacle,
                scene_num,
            ) = task_name.split("-")

            traj_data_f = os.path.join(task, "traj_data.json")
            traj_data = json.load(open(traj_data_f, "r"))

            traj_metadata_f = os.path.join(task, "traj_metadata.pkl")
            traj_data['metadata'] = json.load(open(traj_metadata_f, "r"))
            data.append(traj_data)
            # keys: images, pddl_params, plan, scene, task_id, task_type, turk_annotations

        return data

    def _split_by_semantic_skills(self):
        # create semantic sequences
        semantic_sequences = []
        for data in self.data:
            # data is a json

            anns = data["turk_annotations"][
                "anns"
            ]  # different annotations of the same thing

            # pick a random one
            ann = random.choice(anns)
            task_desc = ann["task_desc"]
            high_level_descs = ann["high_descs"]

            # tokenize language
            # TODO: do we use the task desc?
            tokens = self.tokenizer(high_level_descs, return_tensors="np", padding=True)

            high_level_plan = data["plan"]["high_pddl"]

            low_level_seq = data["plan"]["low_actions"]

            # split actions based on skill alignment
            action_ids = [[]] * len(high_level_descs)
            obj_ids = [[]] * len(high_level_descs)

            for action in low_level_seq:
                high_idx = action["high_idx"]
                action_id = Actions[action["api_action"]["action"]].value
                action_ids[high_idx].append(action_id)

                if "objectId" in action["api_action"]:
                    interact_obj = action["api_action"]["objectId"].split("|")[0]
                    interact_obj_id = constants.OBJECTS.index(interact_obj)
                    obj_ids[high_idx].append(interact_obj_id)
                else:
                    obj_ids[high_idx].append(-1)

            # add a new sequence for each high level action / skill
            for idx in range(len(high_level_descs)):
                sequence = AttrDict(
                    states=np.array(action_ids[idx]),  # TODO:fix
                    actions=np.array(action_ids[idx]),
                    interact_obj=np.array(obj_ids[idx]),
                    lang_token_ids=tokens["input_ids"][idx],
                    lang_attention_mask=tokens["attention_mask"][idx],
                    timesteps=np.arange(len(action_ids[idx])),
                )
                semantic_sequences.append(sequence)
        return semantic_sequences


if __name__ == "__main__":

    hparams = AttrDict(
        alfred_data_dir="/home/anthony/alfred/data/json_2.1.0",
        decoder_model_cls="gpt2",
        load_lang=True,
        chunk_size=512,
    )
    dataset = ALFREDSingleSequenceDataset(hparams=hparams, split="valid_seen")

    import ipdb

    ipdb.set_trace()
    print(len(dataset))
