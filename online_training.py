"""
Implements online training. 

Collect new trajectory and add it to replay buffer in FIFO style, follow stochastic policy 

perform an update step, basically we add the new trajectory to the dataloader
 
we also have a learned entropy term 

we want to sample actions instead of taking the mean 
"""

import json
import torch
import numpy as np
from model.lb_single_seq_decoder import Model as LBSingleSeqDecoder
from evaluate import create_param_grid, load_model_and_env_from_cfg
from hydra.utils import instantiate
from trainer import merge_list_of_cfg
from data.language_behavior import ALFREDLanguageBehaviorDataModule
from utils.eval_utils import discount_cumsum
from pytorch_lightning.trainer.trainer import Trainer
from data.dataset import TrajectoryDataset
from utils.data_utils import collate_fn
from omegaconf import DictConfig, OmegaConf
from data.sampler import ImportanceSamplingBatchSampler
from pytorch_lightning.trainer.supporters import CombinedLoader


def main():
    base_cfg, combined_conf = create_param_grid()
    config = combined_conf[0]

    trainer_cfg = merge_list_of_cfg(base_cfg["trainer"])
    data_cfg = merge_list_of_cfg(base_cfg["data"])
    model_cfg = merge_list_of_cfg(base_cfg["model"])

    # create env and load model
    model, env = load_model_and_env_from_cfg(config)
    model.hparams.update(model_cfg["model_conf"])

    # ignore wandb
    trainer_cfg.logger = trainer_cfg.logger[:-1]
    trainer_cfg.max_epochs = config.online_finetuning_conf.num_steps_per_iter
    trainer_cfg.limit_val_batches = 0
    trainer_cfg.num_sanity_val_steps = 0

    # update config
    config.sampler.config.target_return = 10
    config.sampler.config.return_conditioned = True
    rollout_cls = instantiate(config.sampler, env=env, agent=model)

    # create replay buffer (which is basically the dataset)
    # initialize replay initially with offline demonstrations
    # lb_dm = instantiate(data_cfg, _recursive_=False)
    # lb_dm.prepare_data()

    # # splits/transforms
    # lb_dm.setup(stage="fit")

    # train_ds = lb_dm.datasets["train/behavior"]
    # traj_data = train_ds.jsons_and_keys[0][0]  # keep the same task
    traj_data = json.load(open("sample_task.json", "r"))
    print(f"task: {traj_data['task_type']}")

    # create a trajectory dataset which will store the new rollouts
    trajectory_dataset = TrajectoryDataset(
        data_cfg["data_conf"]["behavior_dataset_cls"]["hparams"]
    )

    for iter_ in range(config.online_finetuning_conf.max_iters):
        trainer = instantiate(trainer_cfg, _recursive_=True)  # reset trainer

        model.eval()
        model = model.cuda()
        rollout_cls.agent = model  # make sure its still the same

        # collect new rollout, using stochastic policy
        # initialize the agent in a random scene

        # stochastic policy to encourage exploration

        episode = rollout_cls.rollout_single_episode(
            traj_data=traj_data, use_means=False
        )

        # create a new chunk and add it to dataloader
        actions = np.stack(episode["action"])
        interact_mask = actions[:, -1] != -1
        valid_mask = np.array(
            [episode["info"][idx]["step_success"] for idx in range(len(actions) - 1)]
        )
        valid_mask = np.concatenate([valid_mask, [True]])

        # need to first split this into semantic seqs
        data = {
            "states": torch.cat(episode["observation"]),
            "actions": actions,
            "timesteps": np.arange(len(actions)),
            "valid_interact_mask": interact_mask & valid_mask,
            "returns_to_go": discount_cumsum(np.array(episode["reward"]), gamma=1),
        }
        new_trajectory = trajectory_dataset.add_masks([[data]])

        # remove oldest batch from offline demonstrations
        # train_ds.chunks = train_ds.chunks[1:]

        # update the datasets variable in the data module
        # lb_dm.datasets["train/behavior"] = train_ds

        # add new rollouts to trajectory dataset
        trajectory_dataset.trajectories.extend(new_trajectory)
        cfg = OmegaConf.create(data_cfg.data_conf["dataloader_cls"])
        batch_sampler = ImportanceSamplingBatchSampler(
            trajectory_dataset.trajectories, cfg["batch_size"]
        )
        del cfg.batch_size
        trajectories_dataloader = CombinedLoader(
            {
                "behavior": instantiate(
                    cfg,
                    dataset=trajectory_dataset,
                    pin_memory=True,
                    worker_init_fn=lambda x: np.random.seed(
                        np.random.randint(65536) + x
                    ),
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn,
                )
            },
            mode="max_size_cycle",
        )

        model.train()

        # perform update step on buffer data
        model.return_conditioned = False
        # trainer.fit(model=model, train_dataloaders=lb_dm.train_dataloader())

        # perform update on rollouts data
        model.return_conditioned = True
        trainer.fit(model=model, train_dataloaders=trajectories_dataloader)

        import ipdb

        ipdb.set_trace()
        # for batch_idx, batch in enumerate(lb_dm.val_dataloader()):
        #     model.eval()
        #     eval_loss = model.validation_step(batch, batch_idx)


if __name__ == "__main__":
    main()
