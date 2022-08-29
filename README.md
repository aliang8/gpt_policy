# GPT Policy

A single decoder transformer model that inputs both state/actions and language as a concatenated sequence. 

```
a1      a2      a3      ...         Open   the    microwave
 |       |       |                    |      |         |
================================ Transformer ================================
 |   |   |   |   |   |   |        |   |      |
s1  a1  s2  a2  s3  a3  s4  ... Open the microwave   <EOS>
```

The transformer learns to autoregressively predict actions given previous states and actions. 
The causal structure of GPT-style transformer decoders enforce causal structure such that the prediction
for actions at time t is conditioned only on states 0->t and actions 0->t-1.

The model also consumes language annotations for action sequences and by way of the how we input
sequences to the model and casual structure, we are able to learn a transition prior between 
skills: p(lang_skill_B | lang_skill_A). 

The model can input three modalities of data: language-only, behavior-only, and paired language-behavior data.
Language-only data will typically come from procedural text where each sentence denotes a skill and 
skills follow each other in a semantically meaningful way. 

Behavioral data can be any offline dataset of demonstrations containing (state, action) pairs. 

Paired behavior data contains behavioral sequences that are annotated with language descriptions. 

### Assumptions about the data:

### Train the model: 
```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --config configs/decoder.yaml
```

### Evaluate the model on new task:
```
# runs rollouts 
python3 multi_eval.py \
    exp_name=[{EXP_NAME}] \
    sampler.config.num_samples=50 \
    sampler.config.max_episode_len=280 \
    eval_config_files=[configs/base/base_eval.yaml]

# generate videos
python3 multi_eval.py \
    exp_name=[{EXP_NAME}] \
    sampler.config.num_samples=50 \
    sampler.config.max_episode_len=280 \
    eval_config_files=[configs/base/base_eval.yaml,configs/custom/video_eval.yaml]
```

### Installation Instructions 

```
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Install d4rl, need custom one
cd ..
git clone https://github.com/kpertsch/d4rl.git
cd d4rl
pip install -e .

pip3 install protobuf==3.19.0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ??

export ALFRED_ROOT=/data/anthony/alfred/
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
```

### Help

```
Check python version
python -c "import torch; print(torch.__version__)"

Check cuda version
python -c "import torch; print(torch.version.cuda)"

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

export ET_ROOT=$(pwd)
export ET_LOGS=$ET_ROOT/logs
export ET_DATA=$ET_ROOT/data
export PYTHONPATH=$PYTHONPATH:$ET_ROOT
export ALFRED_ROOT=/data/anthony/alfred/
```