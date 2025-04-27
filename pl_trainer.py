from oil.datasetup.datasets import split_dataset
from oil.utils.utils import FixedNumpySeed

import lightning.pytorch as pl

import sys
import csv
import io
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch import Tensor

import wandb
import PIL

import numpy as np
from biases.systems.chain_pendulum import ChainPendulum
from biases.systems.rotor import Rotor
from biases.systems.coupled_pendulum import CoupledPendulum
from biases.systems.magnet_pendulum import MagnetPendulum
from biases.systems.gyroscope import Gyroscope
from biases.models.constrained_hnn import CHNN
from biases.models.constrained_lnn import CLNN
from biases.models.hnn import HNN
from biases.models.lnn import LNN, DeLaN
from biases.models.nn import NN, DeltaNN
from biases.datasets import RigidBodyDataset
from biases.systems.rigid_body import rigid_Phi, project_onto_constraints
from typing import List, Optional

from matplotlib import pyplot as plt

class TrajectoryPlotCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 10, n_samples: int = 4,
                 dpi: int = 120):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.dpi = dpi

    def on_validation_epoch_end(self, trainer, pl_module: "DynamicsModel"):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs or trainer.sanity_checking:
            return

        z0, ts, true = _sample_batch(pl_module, self.n_samples, split="val")
        with torch.no_grad():
            pred = pl_module.rollout(z0, ts[0]-ts[0, 0],
                                     tol=pl_module.hparams.tol)

        fig_traj, ax = plt.subplots(figsize=(5, 3), dpi=self.dpi)
        for k in range(self.n_samples):
            ax.plot(true[k, :, 0, 0, 0].cpu(), label=f"true #{k}")
            ax.plot(pred[k, :, 0, 0, 0].cpu(), "--")
        ax.set_title("Validation trajectories"); ax.set_xlabel("t")
        ax.legend(ncol=2, fontsize=6)

        # --- 2. энергия
        E_true = pl_module.true_energy(true)
        E_pred = pl_module.true_energy(pred)
        fig_E, axE = plt.subplots(figsize=(5, 2), dpi=self.dpi)
        axE.plot((E_pred-E_true).abs().mean(0).cpu())
        axE.set_title("|ΔE|"); axE.set_xlabel("t")
        # логируем в W&B
        pl_module.logger.experiment.log(
            {
                "val/traj":      fig_to_img(fig_traj),
                "val/E_abs_err": fig_to_img(fig_E),
            },
            step=epoch,
        )
        plt.close(fig_traj)
        plt.close(fig_E)



def _sample_batch(pl_module: "DynamicsModel", n, split="val"):
    dl = pl_module.val_dataloader() if split == "val" else pl_module.test_dataloader()
    (z0, ts), true = next(iter(dl))
    return z0[:n].to(pl_module.device), ts[:n].to(pl_module.device), true[:n].to(pl_module.device)

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def collect_tensors(field, outputs):
    res = torch.stack([log[field] for log in outputs], dim=0)
    if res.ndim == 1:
        return res
    else:
        return res.flatten(0, 1)


def fig_to_img(fig):
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = wandb.Image(PIL.Image.open(buf))
    return img

def str_to_class(classname):
    import sys
    return getattr(sys.modules[__name__], classname)

def collect_tensors(field, outputs):
    res = torch.stack([log[field] for log in outputs], dim=0)
    return res if res.ndim == 1 else res.flatten(0, 1)

class DynamicsModel(pl.LightningModule):
    def __init__(
        self,
        body_class: str,
        network_class: str,
        *,
        batch_size: int = 128,
        dataset_class: str = "RigidBodyDataset",
        n_train_systems: int = 1000,
        n_train: int = 800,
        n_val: int = 100,
        n_test: int = 100,
        chunk_len: int = 5,
        regen: bool = False,
        seed: int = 0,
        body_args: Optional[List[int]] = None,
        n_hidden: int = 256,
        n_layers: int = 3,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        optimizer_class: str = "AdamW",
        no_lr_sched: bool = False,
        tol: float = 1e-7,
    ):
        super().__init__()
        self._test_outs: list = []
        self.save_hyperparameters()
        self.body = str_to_class(body_class)(*(body_args or []))
        euclidean = network_class not in {"NN", "LNN", "HNN", "DeLaN"}
        self.hparams.euclidean = euclidean 
        self.euclidean = euclidean

        self.hparams["dt"] = self.body.dt
        self.hparams["integration_time"] = self.body.integration_time

        Dataset = str_to_class(dataset_class)
        self.datasets = {
            "train": Dataset(n_systems=n_train_systems, n_subsample=n_train,
                             regen=regen, chunk_len=chunk_len, body=self.body,
                             angular_coords=not euclidean, seed=seed, mode="train"),
            "val":   Dataset(n_systems=n_val, regen=regen, chunk_len=chunk_len,
                             body=self.body, angular_coords=not euclidean,
                             seed=seed+1, mode="val"),
            "test":  Dataset(n_systems=n_test, regen=regen, chunk_len=chunk_len,
                             body=self.body, angular_coords=not euclidean,
                             seed=seed+2, mode="test"),
        }
        # 3) Сеть
        net_cfg = dict(
            dof_ndim=self.body.d if euclidean else self.body.D,
            angular_dims=self.body.angular_dims,
            hidden_size=n_hidden, num_layers=n_layers, wgrad=True
        )
        Net = str_to_class(network_class)
        self.model = Net(G=self.body.body_graph, **net_cfg)

    def forward(self, z0, ts, tol=None, method="rk4"):
        return self.rollout(z0, ts, tol or self.hparams.tol, method)

    def configure_optimizers(self):
        Opt = getattr(torch.optim, self.hparams.optimizer_class)
        opt = Opt(self.parameters(), lr=self.hparams.lr,
                  weight_decay=self.hparams.weight_decay)
        if self.hparams.no_lr_sched:
            return opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=1000, eta_min=0.0)
        return [opt], [sched]

    def _make_loader(self, split: str, shuffle: bool):
        bs = self.hparams.batch_size if split == "train" else len(self.datasets[split])
        return DataLoader(
            self.datasets[split],
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self): return self._make_loader("train", True)
    def val_dataloader(self):   return self._make_loader("val",   False)
    def test_dataloader(self):  return self._make_loader("test",  False)

    def forward(self, z0, ts, tol=None, method="rk4"):
            tol = tol if tol is not None else self.hparams.tol
            return self.rollout(z0, ts, tol, method)
    def rollout(self, z0, ts, tol, method="rk4"):
        if not z0.requires_grad:
            z0 = z0.detach().clone().requires_grad_(True)

        return self.model.integrate(z0, ts, tol=tol, method=method)


    def trajectory_mae(self, pred_zts, true_zts):
        return (pred_zts - true_zts).abs().mean()

    def training_step(self, batch: Tensor, batch_idx: int):
        (z0, ts), zts = batch
        ts = ts[0] - ts[0, 0]
        pred_zs = self.rollout(z0, ts, tol=self.hparams.tol, method="rk4")
        loss = self.trajectory_mae(pred_zs, zts)

        logs = {
            "train/trajectory_mae": loss.detach(),
            "train/nfe": self.model.nfe,
        }
        return {
            "loss": loss,
            "log": logs,
        }

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, integration_factor=self.hparams.chunk_len * self.hparams.dt / self.hparams.integration_time)

    # def validation_epoch_end(self, outputs):
    #     log, save = self._collect_test_steps(outputs)
    #     log = {f"validation/{k}": v for k, v in log.items()}
    #     return {"val_loss": log["validation/trajectory_mae"], "log": log}
    def validation_step(self, batch, batch_idx):
        out = self.test_step(

        batch, batch_idx,
            integration_factor=self.hparams.chunk_len * self.hparams.dt / self.hparams.integration_time
        )
        self.log("validation/trajectory_mae", out["trajectory_mae"],
             on_epoch=True, prog_bar=True)

        for k in ["rel_err_pred_true", "abs_err_pred_true",
                "rel_err_pert_true", "abs_err_pert_true"]:
            self.log(f"validation/{k}", out[k].mean(), on_epoch=True)


    def test_step(self, batch, batch_idx, integration_factor=1.0):
        (z0, ts), _ = batch
        (
            pred_zts,
            true_zts,
            pert_zts,
            rel_err_pred_true,
            abs_err_pred_true,
            rel_err_pert_true,
            abs_err_pert_true,
        ) = self.compare_rollouts(
            z0,
            integration_factor * self.hparams.integration_time,
            self.hparams.dt,
            self.hparams.tol,
        )
        loss = self.trajectory_mae(pred_zts, true_zts)
        pred_zts_true_energy = self.true_energy(pred_zts)
        true_zts_true_energy = self.true_energy(true_zts)
        pert_zts_true_energy = self.true_energy(pert_zts)

        out = {
             "trajectory_mae": loss.detach(),
             "pred_zts": pred_zts.detach(),
             "true_zts": true_zts.detach(),
             "pert_zts": pert_zts.detach(),
             "rel_err_pred_true": rel_err_pred_true.detach(),
             "abs_err_pred_true": abs_err_pred_true.detach(),
             "rel_err_pert_true": rel_err_pert_true.detach(),
             "abs_err_pert_true": abs_err_pert_true.detach(),
             "pred_zts_true_energy": pred_zts_true_energy.detach(),
             "true_zts_true_energy": true_zts_true_energy.detach(),
             "pert_zts_true_energy": pert_zts_true_energy.detach(),
         }
        self._test_outs.append(out)
        return out

    def _collect_test_steps(self, outputs):
        loss = collect_tensors("trajectory_mae", outputs).mean(0).item()
        rel_err_pred_true = (collect_tensors("rel_err_pred_true", outputs)) + 1e-8
        abs_err_pred_true = (collect_tensors("abs_err_pred_true", outputs)) + 1e-8
        rel_err_pert_true = (collect_tensors("rel_err_pert_true", outputs)) + 1e-8
        abs_err_pert_true = (collect_tensors("abs_err_pert_true", outputs)) + 1e-8
        int_rel_err_pred_true = self.integrate_curve(
            rel_err_pred_true.log(), dt=self.hparams.dt
        ).mean(0)
        int_abs_err_pred_true = self.integrate_curve(
            abs_err_pred_true.log(), dt=self.hparams.dt
        ).mean(0)
        int_rel_err_pert_true = self.integrate_curve(
            rel_err_pert_true.log(), dt=self.hparams.dt
        ).mean(0)
        int_abs_err_pert_true = self.integrate_curve(
            abs_err_pert_true.log(), dt=self.hparams.dt
        ).mean(0)

        pred_zts_true_energy = collect_tensors("pred_zts_true_energy", outputs)
        true_zts_true_energy = collect_tensors("true_zts_true_energy", outputs)
        pert_zts_true_energy = collect_tensors("pert_zts_true_energy", outputs)

        int_pred_true_energy = self.integrate_curve(
            pred_zts_true_energy, dt=self.hparams.dt
        ).mean(0)
        int_true_true_energy = self.integrate_curve(
            true_zts_true_energy, dt=self.hparams.dt
        ).mean(0)
        int_pert_true_energy = self.integrate_curve(
            pert_zts_true_energy, dt=self.hparams.dt
        ).mean(0)
        log = {
            "trajectory_mae": loss,
            "int_rel_err_pred_true": int_rel_err_pred_true,
            "int_abs_err_pred_true": int_abs_err_pred_true,
            "int_rel_err_pert_true": int_rel_err_pert_true,
            "int_abs_err_pert_true": int_abs_err_pert_true,
            "int_pred_true_energy": int_pred_true_energy,
            "int_true_true_energy": int_true_true_energy,
            "int_pert_true_energy": int_pert_true_energy,
        }
        pred_zts = collect_tensors("pred_zts", outputs)
        true_zts = collect_tensors("true_zts", outputs)
        pert_zts = collect_tensors("pert_zts", outputs)
        save = {"pred_zts": pred_zts, "true_zts": true_zts, "pert_zts": pert_zts}
        save.update(log)
        return log, save
    def on_test_epoch_start(self):
        self._test_outs.clear()

    def on_test_epoch_end(self):
        log, save = self._collect_test_steps(self._test_outs)
        for k, v in log.items():
            self.log(f"test/{k}", v, prog_bar=False, sync_dist=True)
        self.test_log = save
        self._test_outs.clear()   
    
    def compare_rollouts(
        self, z0: Tensor, integration_time: float, dt: float, tol: float, pert_eps=1e-4
    ):
        prev_device = list(self.parameters())[0].device
        prev_dtype = list(self.parameters())[0].dtype
        ts = torch.arange(0.0, integration_time, dt, device=z0.device, dtype=z0.dtype)
        print("Rolling out model")
        pred_zts = self.rollout(z0, ts, tol, "dopri5")
        bs, Nlong, *rest = pred_zts.shape
        body = self.datasets["test"].body
        if not self.hparams.euclidean:
            z0 = body.body2globalCoords(z0)
            flat_pred = body.body2globalCoords(pred_zts.reshape(bs * Nlong, *rest))

            pred_zts = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])

        perturbation = pert_eps * torch.randn_like(
            z0
        )
        z0_perturbed = project_onto_constraints(
            body.body_graph, z0 + perturbation
        )
        # (bs, n_steps, 2, n_dof, d)
        print("Rolling out true system")
        z0_ = torch.cat([z0, z0_perturbed], dim=0)
        true_zts_pert_zts = body.integrate(z0_, ts, tol=tol)
        true_zts, pert_zts = true_zts_pert_zts.chunk(2, dim=0)

        sq_diff_pred_true = (pred_zts - true_zts).pow(2).sum((2, 3, 4))
        sq_diff_pert_true = (true_zts - pert_zts).pow(2).sum((2, 3, 4))
        sq_sum_pred_true = (pred_zts + true_zts).pow(2).sum((2, 3, 4))
        sq_sum_pert_true = (true_zts + pert_zts).pow(2).sum((2, 3, 4))

        # (bs, n_step)
        rel_err_pred_true = sq_diff_pred_true.div(sq_sum_pred_true).sqrt()
        abs_err_pred_true = sq_diff_pred_true.sqrt()
        rel_err_pert_true = sq_diff_pert_true.div(sq_sum_pert_true).sqrt()
        abs_err_pert_true = sq_diff_pert_true.sqrt()

        self.to(prev_dtype)
        self.to(prev_device)
        return (
            pred_zts,
            true_zts,
            pert_zts,
            rel_err_pred_true,
            abs_err_pred_true,
            rel_err_pert_true,
            abs_err_pert_true,
        )

    def true_energy(self, zs):
        N, T = zs.shape[:2]
        q, qdot = zs.chunk(2, dim=2)
        p = self.body.M @ qdot
        zs = torch.cat([q, p], dim=2)
        energy = self.body.hamiltonian(None, zs.reshape(N * T, -1))
        return energy.reshape(N, T)

    def integrate_curve(self, y, t=None, dt=1.0, axis=-1):
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        return np.trapz(y, t, dx=dt, axis=axis)

    def configure_optimizers(self):
        Opt = getattr(torch.optim, self.hparams.optimizer_class)
        opt = Opt(self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay)

        if self.hparams.no_lr_sched:
            return opt

        max_epochs = getattr(self.trainer, "max_epochs", 1000) or 1000

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=0.0
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
        parser.add_argument(
            "--body-class",
            type=str,
            help="Class name of physical system",
            required=True,
        )
        parser.add_argument(
            "--body-args",
            help="Arguments to initialize physical system separated by spaces",
            nargs="*",
            type=int,
            default=[],
        )
        parser.add_argument(
            "--no-lr-sched",
            action="store_true",
            default=False,
            help="Turn off cosine annealing for learing rate",
        )
        parser.add_argument(
            "--chunk-len",
            type=int,
            default=5,
            help="Length of each chunk of training trajectory",
        )
        parser.add_argument(
            "--dataset-class",
            type=str,
            default="RigidBodyDataset",
            help="Dataset class",
        )
        parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
        parser.add_argument(
            "--n-test", type=int, default=1000, help="Number of test trajectories"
        )
        parser.add_argument(
            "--n-train", type=int, default=800, help="Number of train trajectories"
        )
        parser.add_argument(
            "--n-val", type=int, default=1000, help="Number of validation trajectories"
        )
        parser.add_argument(
            "--network-class",
            type=str,
            help="Dynamics network",
            choices=[
                "NN",
                "DeltaNN",
                "HNN",
                "LNN",
                "DeLaN",
                "CHNN",
                "CLNN",
                "CHLC",
                "CLLC",
            ],
        )
        parser.add_argument(
            "--n-epochs", type=int, default=2000, help="Number of training epochs"
        )
        parser.add_argument(
            "--n-hidden", type=int, default=256, help="Number of hidden units"
        )
        parser.add_argument(
            "--n-layers", type=int, default=3, help="Number of hidden layers"
        )
        parser.add_argument(
            "--n-train-systems", type=int, default=100, help="Number of hidden layers"
        )
        parser.add_argument(
            "--optimizer_class", type=str, default="AdamW", help="Optimizer",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="Seed used to generate dataset",
        )
        parser.add_argument(
            "--tol",
            type=float,
            default=1e-7,
            help="Tolerance for numerical intergration",
        )
        parser.add_argument(
            "--regen",
            action="store_true",
            default=False,
            help="Forcibly regenerate training data",
        )
        parser.add_argument(
            "--weight-decay", type=float, default=1e-4, help="Weight decay",
        )
        return parser


class SaveTestLogCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        if type(trainer.logger) == WandbLogger:
            save_dir = os.path.join(trainer.logger.experiment.dir, "test_log.pt")
            if "test_log" in trainer.callback_metrics:
                torch.save(trainer.callback_metrics["test_log"], save_dir)


def parse_misc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-clip-val", type=float, default=0, help="Threshold for 2-norm of gradient. 0 is off")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug code by running 1 batch of train, val, and test.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help="Directory to save files from this experiment",
    )
    parser.add_argument(
        "--n-epochs-per-val",
        type=int,
        default=100,
        help="Number of training epochs per validation step",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of training GPUs")
    parser.add_argument(
        "--terminate-on-nan",
        action="store_true",
        default=False,
        help="Terminate upon getting nans in loss or parameters at end of each batch"
    )
    parser.add_argument(
        "--tags", type=str, nargs="*", default=None, help="Experiment tags"
    )
    parser.add_argument("--track-grad-norm", type=int, default=-1, help="Log gradient norms")
    parser.add_argument(
        "--wandb-project", type=str, default="", help="Project name for wandb"
    )
    return parser


# if __name__ == "__main__":
#     from pytorch_lightning import Trainer
#     from pytorch_lightning.loggers import WandbLogger
#     # from pytorch_lightning.callbacks import LearningRateLogger

#     from pytorch_lightning.callbacks import LearningRateMonitor

#     parser = parse_misc()
#     parser = DynamicsModel.add_model_specific_args(parser)
#     hparams = parser.parse_args()

#     dynamics_model = DynamicsModel(hparams=hparams)

#     # create experiment directory
#     if hparams.exp_dir == "":
#         exp_dir = os.path.join(
#             os.getcwd(),
#             "experiments",
#             f"{dynamics_model.body.__repr__()}",
#             f"{hparams.network_class}",
#         )
#     else:
#         exp_dir = hparams.exp_dir
#     # Note that this args is shared with the model's hparams so it will be saved
#     vars(hparams).update(exp_dir=exp_dir)
#     if not os.path.exists(exp_dir):
#         os.makedirs(exp_dir)
#         print("Directory ", exp_dir, " Created ")
#     else:
#         print("Directory ", exp_dir, " already exists")

#     logger = WandbLogger(
#         save_dir=exp_dir, project=hparams.wandb_project, log_model=True, tags=hparams.tags
#     )
#     ckpt_dir = os.path.join(
#         logger.experiment.dir, logger.name, f"version_{logger.version}", "checkpoints",
#     )
#     if hparams.no_lr_sched:
#         callbacks = [SaveTestLogCallback()]
#     else:
#         callbacks = [LearningRateMonitor(), SaveTestLogCallback()]
#     vars(hparams).update(
#         check_val_every_n_epoch=hparams.n_epochs_per_val,
#         fast_dev_run=hparams.debug,
#         gpus=hparams.n_gpus,
#         max_epochs=hparams.n_epochs,
#         ckpt_dir=ckpt_dir,
#     )

#     # record human-readable hparams as csv
#     with open(os.path.join(logger.experiment.dir, "args.csv"), "w") as csvfile:
#         args_dict = vars(hparams)  # convert to dict with new copy
#         writer = csv.DictWriter(csvfile, fieldnames=args_dict.keys())
#         writer.writeheader()
#         writer.writerow(args_dict)

#     # trainer = Trainer.from_argparse_args(hparams, callbacks=callbacks, logger=logger)
#     # trainer = Trainer(**vars(hparams), callbacks=callbacks, logger=logger)
#     # assume you already did something like:
#     trainer_kwargs = vars(hparams).copy()

#     # remove any keys Trainer doesn’t accept:
#     for bad in ("debug", "exp_dir", "n_epochs_per_val", "n_gpus", 
#                 "terminate_on_nan", "tags", "track_grad_norm", "wandb_project",
#                 "batch_size", "body_class", "body_args"):
#         trainer_kwargs.pop(bad, None)

#     trainer = Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)



#     trainer.fit(dynamics_model)

#     with torch.no_grad():
#         trainer.test()

if __name__ == "__main__":
    from lightning.pytorch.cli import LightningCLI

    # LightningCLI(save_config_kwargs={"overwrite": True})
    LightningCLI(DynamicsModel)