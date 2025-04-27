#!/usr/bin/env bash
set -e

python3 test.py \
    test \
    --ckpt_path=lightning_logs/version_0/checkpoints/epoch=49-step=350.ckpt \
    --model.body_class=Gyroscope \
    --model.network_class=CHNN \
    --model.n_layers=8 \
    --model.n_hidden=256 \
    --trainer.inference_mode False\
    --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project=gyro-demo \
    --trainer.callbacks+=callbacks.SaveTestLogCallback
