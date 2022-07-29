# Train
CUDA_VISIBLE_DEVICES=3 python3 -m ipdb -c continue  trainer.py \
    --config configs/base_train.yaml

# Evaluate
CUDA_VISIBLE_DEVICES=1 python3 -m ipdb -c continue  evaluate.py \
    --trainer-config-file configs/trainer.yaml \
    --eval-config-file configs/base_eval.yaml


CUDA_VISIBLE_DEVICES=2 python3 -m ipdb -c continue  evaluate.py \
    --eval-config-file configs/eval_rollout_rewards.yaml
