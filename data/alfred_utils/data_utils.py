import os
import copy
import json


def update_log(dout, stage, update, **kwargs):
    """
    updating a method json for monitoring on Alex's machine
    """
    assert update in ("increase", "rewrite")
    info_path = os.path.join(dout, "info.json")
    assert os.path.exists(info_path)
    with open(info_path) as f:
        info_dicts = json.load(f)
    info_dict = copy.deepcopy([el for el in info_dicts if el["stage"] == stage][-1])
    # update the values
    for key, value in kwargs.items():
        assert key in info_dict
        new_value = value + info_dict[key] if update == "increase" else value
        info_dict[key] = new_value
    # decide what to do with the list with updated values
    if info_dicts[-1]["stage"] == stage:
        # rewrite the values
        info_dicts[-1] = info_dict
    else:
        # append a new list element
        info_dicts.append(info_dict)
    # dump to the disk
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)


def save_log(dout, progress, total, stage, **kwargs):
    """
    logging a method json for besteffort mode and jobs monitoring on Alex's machine
    """
    info_path = os.path.join(dout, "info.json")
    info_dicts = []
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info_dicts = json.load(f)
    info_dict = {"stage": stage, "progress": progress, "total": total}
    info_dict.update(kwargs)
    info_dicts.append(info_dict)
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)
