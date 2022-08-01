# Train
CUDA_VISIBLE_DEVICES=3 python3 -m ipdb -c continue  trainer.py \
    --config configs/base/base_train.yaml

# Evaluate
# multi-gpu eval
# no spaces between list items
python3 -m ipdb -c continue multi_eval.py \
    exp_name=[transformerBC_paired_only,transformerBC_paired_language_pred_progress] \
    sampler.config.num_samples=2 \
    eval_config_files=[configs/base/base_eval.yaml]


python3 -m ipdb -c continue multi_eval.py \
    exp_name=[transformerBC_paired_language_pred_progress] \
    sampler.config.num_samples=20 \
    eval_config_files=[configs/base/base_eval.yaml] \
    debug=True