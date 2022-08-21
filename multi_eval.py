import os
import copy
import random
import torch
import glob
import multiprocessing
from functools import partial

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from pytorch_lightning.utilities.cli import LightningCLI
from evaluate import load_model_and_env_from_cfg, create_param_grid

"""
Runs eval for different train models across separate GPUs
"""

total_gpu_num = 4
max_process_per_gpu = 2
used_gpu_list = multiprocessing.Manager().list([0] * total_gpu_num)
lock = multiprocessing.Lock()


def run_eval(conf):
    model, env = load_model_and_env_from_cfg(conf)

    # create rollout helper
    rollout = instantiate(conf.sampler, env=env, agent=model)

    # collect trajectories
    episodes = rollout.rollout_multi_episode()
    return episodes


def multi_gpu_testing_wrapper(eval_cfg, index, gpu_id=None, available_gpu_num=1):
    # GPU assignment
    lock.acquire()
    if gpu_id is None:
        for i in range(available_gpu_num):
            if used_gpu_list[i] < max_process_per_gpu:
                gpu_id = i
                break

    used_gpu_list[gpu_id] += 1
    lock.release()
    torch.cuda.set_device(gpu_id)
    print(
        f"testing   input {index} on GPU {gpu_id}. Overall GPU usages: ",
        list(used_gpu_list),
    )

    episodes, info = run_eval(eval_cfg)
    torch.cuda.empty_cache()

    # release GPU
    lock.acquire()
    used_gpu_list[gpu_id] -= 1
    lock.release()
    print(
        f"releasing input {index} on GPU {gpu_id}. Overall GPU usages: ",
        list(used_gpu_list),
    )

    # return output
    return episodes, info, gpu_id, os.getpid()


def main():
    # ================ CREATE PARAMETER GRID ================
    base_cfg, all_configs = create_param_grid()

    # setup GPU
    available_gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert available_gpu_num <= total_gpu_num
    print(available_gpu_num)

    if "debug" in base_cfg and base_cfg.debug:
        episodes, info = run_eval(all_configs[0])
        num_episodes = len(episodes)
        print(f"Collected {num_episodes} episodes")
        print(info)
    else:
        output = {}

        def mycallback(arg, cfg):
            output[cfg.sampler.config.exp_name] = arg[1]

        pool = multiprocessing.Pool(available_gpu_num * max_process_per_gpu)

        # create separate process for each eval job
        for i, conf in enumerate(all_configs):
            pool.apply_async(
                multi_gpu_testing_wrapper,
                args=(conf, i, None, available_gpu_num),
                callback=partial(mycallback, cfg=conf),
            )

        pool.close()
        pool.join()
        print(output)
        print("All subprocesses done.")


if __name__ == "__main__":
    main()
