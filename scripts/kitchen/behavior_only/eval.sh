EXP_NAME=behavior_only
# python3 -m ipdb -c continue multi_eval.py \
#     exp_name=[$EXP_NAME] \
#     sampler.config.num_samples=50 \
#     sampler.config.video_prefix=behavior_only \
#     eval_config_files=[configs/base/base_eval.yaml] \
#     debug=True

# python3 -m ipdb -c continue multi_eval.py \
#     exp_name=[$EXP_NAME] \
#     sampler.config.num_samples=50 \
#     sampler.config.video_prefix=behavior_only \
#     env.config.name=kitchen-all-tasks-v0 \
#     eval_config_files=[configs/base/base_eval.yaml] \
#     debug=True

python3 -m ipdb -c continue multi_eval.py \
    exp_name=[$EXP_NAME] \
    sampler.config.num_samples=10 \
    sampler.config.video_prefix=behavior_only_199 \
    env.config.name=kitchen-all-tasks-v0 \
    eval_config_files=[configs/base/base_eval.yaml,configs/custom/video_eval.yaml] \
    debug=True
