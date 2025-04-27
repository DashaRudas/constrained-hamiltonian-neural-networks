#!/usr/bin/env bash
set -e

python3 pl_trainer.py fit \
  --model.network_class CHNN \
  --model.body_class   Gyroscope \
  --model.n_train_systems 1000 \
  --model.n_train 800 \
  --model.n_val   100 \
  --model.n_test  100 \
  --model.chunk_len   5 \
  --model.batch_size  128 \
  \
  --trainer.accelerator cpu \
  --trainer.devices     1 \
  --trainer.max_epochs  50 \
  \
  --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
  --trainer.logger.init_args.project gyro-demo \
  --trainer.logger.init_args.tags   '[train,viz]' \
  \
  --trainer.callbacks.0.class_path lightning.pytorch.callbacks.LearningRateMonitor \
  \
  --trainer.callbacks.1.class_path callbacks.SaveTestLogCallback \
  \
  --trainer.callbacks.2.class_path callbacks.TrajectoryPlotCallback \
  --trainer.callbacks.2.init_args.every_n_epochs 5 \
  --trainer.callbacks.2.init_args.n_samples      4
