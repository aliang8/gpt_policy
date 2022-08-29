python3 -m ipdb -c continue trainer.py \
    trainer=[configs/base/trainer.yaml,configs/kitchen/single_seq/trainer.yaml] \
    data=configs/kitchen/single_seq/data.yaml \
    model=[configs/base/decoder_model.yaml,configs/kitchen/single_seq/model.yaml]
