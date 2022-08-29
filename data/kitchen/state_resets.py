import gym
import d4rl
import numpy as np
import pickle
from data.kitchen import KitchenDataset
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

env = gym.make("kitchen-mixed-v0")
dataset = env.get_dataset()
hparams = AttrDict(discretize_actions=False, load_lang=False)
kitchen_dataset = KitchenDataset(hparams, dataset)

LS_INDX = 2

# collect states right before light switch
# also save the timesteps that they occur
ls_init_states = []
timesteps = []

for seq in kitchen_dataset.data:
    states = seq["states"]
    skills = seq["skills"]

    indices = np.where(skills == LS_INDX)[0][-20:-15]
    timesteps.append(indices)
    ls_init_states.append(states[indices])

ls_init_states = np.concatenate(ls_init_states)
timesteps = np.concatenate(timesteps)

pickle.dump(
    {"states": ls_init_states, "timesteps": timesteps},
    open("/data/anthony/gpt_policy/data/kitchen/init_states.pkl", "wb"),
)
