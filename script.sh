#!/usr/bin/env bash
set -e

python3 train.py fit \
  --model.body_class=MagneticTrap \
  --model.network_class=LNN \
  --model.dataset_class=MagneticTrapDataset \
  --model.n_train_systems=100 --model.n_train=80 --model.n_val=10 --model.n_test=10 \
      --model.chunk_len=5 --model.batch_size=128 \
      --trainer.accelerator=cpu --trainer.devices=1 --trainer.max_epochs=50 \
      --trainer.check_val_every_n_epoch=5 \
      --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
      --trainer.logger.init_args.project=magn-demo \
      --trainer.logger.init_args.tags='[\"train\",\"viz\"]' \