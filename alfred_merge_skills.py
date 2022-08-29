import os
import copy
import json
import numpy as np
import random
from tqdm import tqdm

# generate eval_skills for a specific scene and seed

# THIS FILE GENERATES NEW SKILLS WITH GOTOLOCATION MERGED

process_set = "valid_seen"  # "train, valid_seen, valid_unseen"

data_path = "/home/anthony/alfred/json_2.1.0"
splits_path = "/home/anthony/alfred/splits/oct21.json"
save_path = "/home/anthony/alfred/json_2.1.0_merge_goto"

with open(splits_path) as f:
    splits = json.load(f)


def pass_file(traj_data):
    plan = traj_data["plan"]["high_pddl"]
    for skill in plan:
        if skill["discrete_action"]["action"] == "GotoLocation":
            return False
    return True


def get_goto_idx(traj_data):
    plan = traj_data["plan"]["high_pddl"]
    goto_idx = []
    for skill in plan:
        if skill["discrete_action"]["action"] == "GotoLocation":
            goto_idx.append(skill["high_idx"])
    return goto_idx


def merge_goto_api_action(traj_data, goto_idx_list):

    # match low level action
    for action in traj_data["plan"]["low_actions"]:
        # merge goto skill to the skill in front of it it
        if action["high_idx"] in goto_idx_list:
            action["high_idx"] += 1
        # shift high_idx without goto skill index
        dis_num = 0
        for goto_id in goto_idx_list:
            if action["high_idx"] > goto_id:
                dis_num += 1
        action["high_idx"] -= dis_num

    for skill in traj_data["plan"]["high_pddl"]:
        # remove goto location skill
        if skill["discrete_action"]["action"] == "GotoLocation":
            traj_data["plan"]["high_pddl"].remove(skill)

    for skill in traj_data["plan"]["high_pddl"]:
        # shift high_idx without goto skill index
        dis_num = 0
        for goto_id in goto_idx_list:
            if skill["high_idx"] > goto_id:
                dis_num += 1
        skill["high_idx"] -= dis_num

    for ann in traj_data["turk_annotations"]["anns"]:
        # choose annotation without goto skill
        new_anns = []

        for goto_id in range(len(ann["high_descs"])):
            if goto_id not in goto_idx_list:
                new_anns.append(ann["high_descs"][goto_id])
        ann["high_descs"] = new_anns
        # print(len(traj_data["plan"]["high_pddl"]) - len(new_anns))
        assert (len(traj_data["plan"]["high_pddl"]) - len(new_anns) <= 2) and (
            len(traj_data["plan"]["high_pddl"]) - len(new_anns) >= 0
        )
        # ann["high_descs"] = [
        #     desc
        #     for desc in ann["high_descs"]
        #     if ann["high_descs"].index(desc) not in goto_idx_list
        # ]

    return traj_data


# trails = splits[process_set]
# a = 0
# for trail in tqdm(trails):
#     traj = trail["task"]
#     # traj_path = os.path.join(
#     #     data_path, process_set, traj, "augmented_traj_data.json"
#     # )  # if you didn't augment the dataset, you can use "traj_data.json"
#     traj_path = os.path.join(
#         data_path, process_set, traj, "traj_data.json"
#     )  # if you didn't augment the dataset, you can use "traj_data.json"

#     if not os.path.exists(traj_path):
#         a += 1
#         print(a)
#         continue

#     folder_path = os.path.join(save_path, process_set, traj)
#     os.makedirs(folder_path, exist_ok=True)
#     # traj_new_path = os.path.join(
#     #     save_path, process_set, traj, "augmented_traj_data_new.json"
#     # )  # "traj_data_new.json"
#     traj_new_path = os.path.join(
#         save_path, process_set, traj, "traj_data_new.json"
#     )  # "traj_data_new.json"

#     if trails[0]["repeat_idx"] == 0:
#         with open(traj_path) as f:
#             traj_data = json.load(f)

#         pass_ = pass_file(traj_data)

#         if not pass_:
#             goto_idx_list = get_goto_idx(traj_data)
#             traj_data = merge_goto_api_action(traj_data, goto_idx_list)

#             with open(traj_new_path, "w") as f:
#                 json.dump(traj_data, f, indent=4)


trails = splits[process_set]

for trail in tqdm(trails):
    traj = trail["task"]
    traj_path = os.path.join(data_path, process_set, traj, "traj_data.json")
    traj_path_new = os.path.join(save_path, process_set, traj, "traj_data_new.json")

    with open(traj_path) as f:
        traj_data = json.load(f)

    with open(traj_path_new) as f:
        traj_data_new = json.load(f)

    actions = traj_data["plan"]["low_actions"]
    print("old")
    print([action["high_idx"] for action in actions])

    new_actions = traj_data_new["plan"]["low_actions"]
    print("new")
    print([action["high_idx"] for action in new_actions])

    import ipdb

    ipdb.set_trace()
