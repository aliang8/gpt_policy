import torch
import queue
import filelock
from termcolor import colored
import numpy as np

from hydra.utils import instantiate
from envs.alfred_env import ALFREDEnv

# from evals.rollout import Rollout
from model.modules.alfred_visual_enc import FeatureExtractor


def disable_lock_logs():
    lock_logger = filelock.logger()
    lock_logger.setLevel(30)


def load_object_predictor(args):
    if args.object_predictor is None:
        return None

    return FeatureExtractor(
        archi="maskrcnn",
        device=args.device,
        checkpoint=args.object_predictor,
        load_heads=True,
    )


def worker_loop(
    args,
    model,
    object_predictor,
    dataset,
    trial_queue,
    log_queue,
    logger,
    cuda_device=None,
):
    """
    evaluation loop
    """
    if cuda_device:
        torch.cuda.set_device(cuda_device)
        args.device = "cuda:{}".format(cuda_device)

    # start THOR
    env = instantiate(args.env)

    # master may ask to evaluate different models

    model_path_loaded = None
    sampler = Rollout(args, env, model)

    import ipdb

    ipdb.set_trace()
    if args.num_workers == 0:
        num_success, num_trials_done, num_trials = 0, 0, trial_queue.qsize()
    try:
        while True:
            trial_uid, dataset_idx, model_path = trial_queue.get(timeout=3)

            # if model_path != model_path_loaded:
            #     if model_path_loaded is not None:
            #         del model, extractor
            #         torch.cuda.empty_cache()

            dataset.vocab_translate = model.vocab_out
            model_path_loaded = model_path

            log_entry = sampler.rollout_single_episode_binary(
                trial_uid,
                dataset_idx,
            )

            if (args.debug or args.num_workers == 0) and "success" in log_entry:
                if "subgoal_action" in log_entry:
                    trial_type = log_entry["subgoal_action"]
                else:
                    trial_type = "full task"

                print(
                    colored(
                        "Trial {}: {} ({})".format(
                            trial_uid,
                            "success" if log_entry["success"] else "fail",
                            trial_type,
                        ),
                        "green" if log_entry["success"] else "red",
                    )
                )

            if args.num_workers == 0 and "success" in log_entry:
                num_trials_done += 1
                num_success += int(log_entry["success"])
                print(
                    "{:4d}/{} trials are done (current SR = {:.1f})".format(
                        num_trials_done, num_trials, 100 * num_success / num_trials_done
                    )
                )

            log_queue.put((log_entry, trial_uid, model_path))

    except queue.Empty:
        pass
    # stop THOR
    env.stop()


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum
