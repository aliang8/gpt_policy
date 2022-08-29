EXP_NAME=with_paired_lang_pred
python3 -m ipdb -c continue multi_eval.py \
    exp_name=[$EXP_NAME] \
    sampler.config.num_samples=50 \
    sampler.config.video_prefix=with_paired_lang_pred \
    sampler.config.predict_lang=True \
    eval_config_files=[configs/base/base_eval.yaml,configs/language_only/generation.yaml] \
    debug=True

# python3 -m ipdb -c continue multi_eval.py \
#     exp_name=[$EXP_NAME] \
#     sampler.config.num_samples=50 \
#     sampler.config.video_prefix=with_paired_lang_pred \
#     sampler.config.predict_lang=True \
#     env.config.name=kitchen-all-tasks-v0 \
#     eval_config_files=[configs/base/base_eval.yaml,configs/language_only/generation.yaml] \
#     debug=True

# python3 -m ipdb -c continue multi_eval.py \
#     exp_name=[$EXP_NAME] \
#     sampler.config.num_samples=20 \
#     sampler.config.video_prefix=with_paired_lang_pred \
#     sampler.config.predict_lang=True \
#     env.config.name=kitchen-all-tasks-v0 \
#     eval_config_files=[configs/base/base_eval.yaml,configs/language_only/generation.yaml,configs/custom/video_eval.yaml] \
#     debug=True
