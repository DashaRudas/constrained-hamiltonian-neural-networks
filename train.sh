#!/usr/bin/env bash
set -e

python3 train.py fit \
      --model.body_class=Gyroscope \
      --model.network_class=CHNN \
      --model.n_train_systems=1000 --model.n_train=800 --model.n_val=100 --model.n_test=100 \
      --model.chunk_len=5 --model.batch_size=128 \
      --trainer.accelerator=cpu --trainer.devices=1 --trainer.max_epochs=50 \
      --trainer.check_val_every_n_epoch=5 \
      --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
      --trainer.logger.init_args.project=gyro-demo \
      --trainer.logger.init_args.tags='[\"train\",\"viz\"]' \
      --trainer.callbacks+=callbacks.SaveTestLogCallback \
        --trainer.callbacks+=callbacks.VisualizeTrajectoriesCallback