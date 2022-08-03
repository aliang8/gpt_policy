# Train
CUDA_VISIBLE_DEVICES=3 python3 -m ipdb -c continue  trainer.py \
    trainer=configs/base/trainer.yaml \
    data=configs/base/data.yaml \
    model=configs/base/decoder_model.yaml

CUDA_VISIBLE_DEVICES=3 python3 -m ipdb -c continue  trainer.py \
    trainer=[configs/base/trainer.yaml,configs/language_only/trainer.yaml] \
    data=[configs/base/data.yaml,configs/language_only/data.yaml] \
    model=configs/base/decoder_model.yaml

# Evaluate
# multi-gpu eval
# no spaces between list items
python3 -m ipdb -c continue multi_eval.py \
    exp_name=[transformerBC_paired_language_pred_done_plus_timestep_emb] \
    sampler.config.num_samples=50 \
    eval_config_files=[configs/base/base_eval.yaml] \

python3 -m ipdb -c continue multi_eval.py \
    exp_name=[transformerBC_paired_language_pred_progress] \
    sampler.config.num_samples=5 \
    sampler.config.max_episode_len=280 \
    sampler.config.self_prompting=True \
    sampler.config.video_prefix=prompting_mtbs \
    sampler.config.prompt="open the microwave. toggle the top burner. toggle the light switch." \
    eval_config_files=[configs/base/base_eval.yaml,configs/custom/video_eval.yaml] \
    debug=True

# test language, make sure language-only prompting works
python3 -m ipdb -c continue scripts/test_language_gen.py \
    exp_name=[transformerBC_paired_language_pred_progress_plus_timestep_emb] \
    sampler.config.num_samples=1 \
    eval_config_files=[configs/base/base_eval.yaml]

# custom
python3 -m ipdb -c continue multi_eval.py \
    exp_name=[transformerBC_paired_language_pred_progress_plus_timestep_emb] \
    sampler.config.num_samples=50 \
    env.config.name=kitchen-mlsh-v0 \
    sampler.config.self_prompting=True \
    eval_config_files=[configs/base/base_eval.yaml] \
