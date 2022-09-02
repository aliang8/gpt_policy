import os
import time
import tqdm
import glob
import json
import lmdb
import torch
import random
import pickle
import warnings
from enum import Enum
import numpy as np
from data.dataset import (
    BaseDataset,
    SingleSequenceDataset,
    SingleSequenceBinaryDataset,
)

from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from utils.lang_utils import (
    get_tokenizer,
    add_start_and_end_token,
    add_start_and_end_str,
)
from typing import Optional, Dict, List, Any

# from ai2thor.interact import DefaultActions
import sys

sys.path.append("/data/anthony/alfred")
import gen.constants as constants

from utils.logger_utils import get_logger

logger = get_logger("alfred_data")


class ALFREDDataset(BaseDataset):
    def __init__(self, hparams: Dict, *args, **kwargs):
        self.hparams = hparams
        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        start = time.time()
        logger.info("loading dataset")
        self.jsons_and_keys = self.load_data()
        logger.info(f"took {time.time() - start} time to load")

        dataset_size = len(self.jsons_and_keys)
        logger.info(f"alfred {self.hparams['partition']} dataset size = {dataset_size}")

        with open(os.path.join(self.hparams.data_dir, "params.json"), "r") as f_params:
            self.dataset_info = json.load(f_params)

        self.vocab_action = torch.load(
            os.path.join(self.hparams.data_dir, "data.vocab")
        )["action_low"]
        logger.info(f"num actions: {len(self.vocab_action)}")

        self.vocab_obj = torch.load(
            os.path.join(self.hparams.data_dir, "obj_cls.vocab")
        )
        logger.info(f"num objects: {len(self.vocab_obj)}")

    # def encode_list_objects(self, list_objs, max_visible_objects):
    #     obj_encodings = []

    #     for obj in list_objs:
    #         obj_encoding = self.encode_object(obj)
    #         obj_encodings.append(obj_encoding)

    #     obj_encodings = np.stack(obj_encodings)
    #     num_objs, emb_dim = obj_encodings.shape
    #     obj_encodings_pad = np.zeros((max_visible_objects, emb_dim))
    #     obj_encodings_pad[:num_objs] = obj_encodings

    #     return obj_encodings_pad

    # def encode_object(self, obj_metadata):
    #     # object encoding: name + position + rotation + one_hot_binary_properties
    #     # name = obj_metadata["name"]
    #     name = obj_metadata["objectType"]
    #     # tokenize object type
    #     # name_tokens = self.obj2tok[name]
    #     # name_tokens_pad = np.zeros((self.max_obj_tokens))
    #     # name_tokens_pad[: len(name_tokens)] = name_tokens

    #     name_tokens_pad = np.array([constants.OBJECTS.index(name)])

    #     position = obj_metadata["position"]
    #     position_vec = np.array([position["x"], position["y"], position["z"]])
    #     rotation = obj_metadata["rotation"]
    #     rotation_vec = np.array([rotation["x"], rotation["y"], rotation["z"]])
    #     distance = obj_metadata["distance"]  # some floating point value
    #     distance = np.array([distance])

    #     # 23 dimensions
    #     state_vec = np.array(
    #         [obj_metadata[k] for k in self.boolean_properties], dtype=np.int32
    #     )

    #     # handle parentReceptacle, receptacleObjectIds, and ObjectTemperature
    #     if obj_metadata["parentReceptacle"]:
    #         pass

    #     if obj_metadata[
    #         "receptacleObjectIds"
    #     ]:  # this should go into the relationship graph
    #         pass
    #         # print(f"{obj_metadata['objectId']} contains: ")

    #     # 37 dimensions if we encode name as tokens
    #     obj_encoding = np.concatenate(
    #         [name_tokens_pad, position_vec, rotation_vec, distance, state_vec]
    #     )
    #     return obj_encoding

    def load_data(self, feats=True, jsons=True):
        """
        load data
        """
        # do not open the lmdb database open in the main process, do it in each thread
        if feats:
            self.feats_lmdb_path = os.path.join(
                self.hparams.data_dir, self.hparams.partition, "feats"
            )
            assert os.path.exists(self.feats_lmdb_path)

        jsons_and_keys = []

        # load jsons with pickle and parse them
        if jsons:
            jsons_file = os.path.join(
                self.hparams.data_dir, self.hparams.partition, "jsons.pkl"
            )
            assert os.path.exists(jsons_file)
            logger.info(f"loading data from: {jsons_file}")

            with open(jsons_file, "rb") as jsons_file:
                jsons = pickle.load(jsons_file)

            logger.info(f"num jsons: {len(jsons)}")

            count = 0
            for idx in range(len(jsons)):
                # if self.hparams.debug and count > 100:
                #     break

                key = "{:06}".format(idx).encode("ascii")
                if key in jsons:
                    task_jsons = jsons[key]
                    # pick a random one
                    json = random.choice(task_jsons)
                    # for json in task_jsons:
                    if json["split"] != self.hparams.partition:
                        continue

                    # compatibility with the evaluation
                    if "task" in json and isinstance(json["task"], str):
                        pass
                    else:
                        json["task"] = "/".join(json["root"].split("/")[-3:-1])
                    # add dataset idx and partition into the json
                    # json["dataset_name"] = self.name
                    jsons_and_keys.append((json, key))
                    # if the dataset has script annotations, do not add identical data
                    if len(set([str(j["ann"]["instr"]) for j in task_jsons])) == 1:
                        break
                count += 1
        return jsons_and_keys

    def load_frames(self, key):
        """
        load image features from the disk
        """
        if not hasattr(self, "feats_lmdb"):
            self.feats_lmdb, self.feats = self.load_lmdb(self.feats_lmdb_path)
        feats_bytes = self.feats.get(key)
        feats_numpy = np.frombuffer(feats_bytes, dtype=np.float32).reshape(
            self.dataset_info["feat_shape"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = torch.tensor(feats_numpy)
        return frames

    def load_lmdb(self, lmdb_path):
        """
        load lmdb (should be executed in each worker on demand)
        """
        database = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=252,
        )
        cursor = database.begin(write=False)
        return database, cursor

    def __del__(self):
        """
        close the dataset
        """
        if hasattr(self, "feats_lmdb"):
            self.feats_lmdb.close()
        if hasattr(self, "masks_lmdb"):
            self.masks_lmdb.close()


class SemanticSkillsALFREDDataset(ALFREDDataset):
    def __init__(self, hparams: Dict, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        # self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        # obj_tokens = self.tokenizer(constants.OBJECTS, return_tensors="np")["input_ids"]

        # self.obj2tok = {
        #     obj_type: np.array(obj_tokens[i])
        #     for i, obj_type in enumerate(constants.OBJECTS)
        # }
        # self.max_obj_tokens = max([len(tokens) for tokens in obj_tokens])

    def has_interaction(self, action):
        """
        check if low-level action is interactive
        """
        non_interact_actions = [
            "MoveAhead",
            "Rotate",
            "Look",
            "<<stop>>",
            "<<pad>>",
            "<<seg>>",
        ]
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True

    def _split_by_semantic_skills(self):
        # create semantic sequences
        semantic_sequences = []

        for idx, json_and_key in tqdm.tqdm(enumerate(self.jsons_and_keys)):
            task_json, key = json_and_key
            image_feats = self.load_frames(key)

            anns = task_json["turk_annotations"][
                "anns"
            ]  # different annotations of the same thing

            # pick a random one
            ann = random.choice(anns)
            task_desc = ann["task_desc"]
            high_level_descs = ann["high_descs"]
            high_level_descs = add_start_and_end_str(high_level_descs)

            # tokenize language
            # TODO: do we use the task desc?
            tokens = self.tokenizer(high_level_descs, return_tensors="np", padding=True)

            high_level_plan = task_json["plan"]["high_pddl"]

            # split actions based on skill alignment
            # last action is a <<stop>> action
            action_ids = [
                [a["action"] for a in a_list]
                for a_list in task_json["num"]["action_low"]
            ]

            valid_interact = [
                a["valid_interact"] for a in sum(task_json["num"]["action_low"], [])
            ]

            num_semantic_skills = len(high_level_descs) + 1
            obj_ids = [[] for i in range(num_semantic_skills)]

            # not sure why the alignment isn't correct here
            num_steps = 0
            for i, a_list in enumerate(task_json["num"]["action_low"][:-1]):
                for action in a_list:
                    high_idx = action["high_idx"]

                    action = task_json["plan"]["low_actions"][num_steps]
                    if valid_interact[num_steps]:
                        obj_key = (
                            "receptacleObjectId"
                            if "receptacleObjectId" in action["api_action"]
                            else "objectId"
                        )
                        object_class = action["api_action"][obj_key].split("|")[0]
                        interact_obj_id = self.vocab_obj.word2index(object_class)
                        obj_ids[high_idx].append(interact_obj_id)
                    else:
                        obj_ids[high_idx].append(
                            -1
                        )  # 0 is the index for no object interacted

                    num_steps += 1

            obj_ids[-1].append(-1)
            low_to_high = np.array(task_json["num"]["low_to_high_idx"])
            stop_tok = self.tokenizer(
                add_start_and_end_str(["stop"])[0], return_tensors="np"
            )
            timesteps = np.arange(num_steps + 1)  # plus 1 for the stop action

            # add a new sequence for each high level action / skill
            for s_idx in range(num_semantic_skills):
                mask = np.where(low_to_high == s_idx)
                states = image_feats[mask]
                actions = np.stack([action_ids[s_idx], obj_ids[s_idx]]).transpose(1, 0)

                if s_idx >= len(tokens["input_ids"]):
                    lang_token_ids = stop_tok["input_ids"][0]  # for the <<stop>> token
                    attn_mask = stop_tok["attention_mask"][0]
                else:
                    lang_token_ids = tokens["input_ids"][s_idx]
                    attn_mask = tokens["attention_mask"][s_idx]

                sequence = AttrDict(
                    states=states,
                    actions=actions,
                    interact_obj=np.array(obj_ids[s_idx]),
                    valid_interact=np.array(valid_interact)[mask],
                    lang_token_ids=lang_token_ids,
                    lang_attention_mask=attn_mask,
                    timesteps=timesteps[mask],
                    skills=low_to_high[mask],
                )

                dones = self._add_done_info(sequence)
                sequence.done = dones
                sequence.first_states = np.zeros_like(sequence.done)
                sequence.first_states[0] = 1

                semantic_sequences.append(sequence)

        return semantic_sequences


class ALFREDSingleSequenceDataset(SingleSequenceDataset, SemanticSkillsALFREDDataset):
    def __init__(self, hparams: Dict, dataset: List, *args, **kwargs):
        self.hparams = hparams
        SemanticSkillsALFREDDataset.__init__(self, hparams, *args, **kwargs)
        SingleSequenceDataset.__init__(self, *args, **kwargs)


if __name__ == "__main__":

    hparams = AttrDict(
        data_dir="/misery/anthony/alfred_orig_lmdb_2.1.0",
        # data_dir="/data/anthony/ET/data/lmdb_full",
        ET_ROOT="/data/anthony/ET",
        decoder_model_cls="gpt2",
        load_lang=True,
        chunk_size=512,
        partition="train",
        load_frac_done=False,
        debug=True,
        return_conditioned=True,
        input_format="v1",
    )
    dataset = ALFREDSingleSequenceDataset(hparams=hparams, dataset=None)

    for data in dataset:
        for k, v in data.items():
            if v is None:
                import ipdb

                ipdb.set_trace()
            assert v is not None

    import ipdb

    ipdb.set_trace()

    print(len(dataset))
