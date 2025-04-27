# callbacks.py
import os, io, torch, wandb, matplotlib.pyplot as plt
import lightning.pytorch as pl
from pl_trainer import fig_to_img, collect_tensors   # утилиты из main-файла

# ─── SaveTestLogCallback ────────────────────────────────────────────────────
class SaveTestLogCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        # test_log теперь лежит внутри pl_module, а не в callback_metrics
        if not getattr(pl_module, "test_log", None):
            return

        if not hasattr(trainer.logger, "experiment"):      # нет WandB – пропускаем
            return

        path = os.path.join(trainer.logger.experiment.dir, "test_log.pt")
        torch.save(pl_module.test_log, path)
        
# ─── VisualizeTrajectoriesCallback ──────────────────────────────────────────
class VisualizeTrajectoriesCallback(pl.Callback):
    """
    Каждые `every_n_epochs` эпох рисует:
      • истинные vs предсказанные координаты
      • |ΔE| (абсолютную ошибку энергии)
    и логирует в Weights & Biases (если WandB-логгер активен).
    """
    def __init__(self, every_n_epochs: int = 10, n_samples: int = 4, dpi: int = 120):
        super().__init__()
        self.every_n_epochs, self.n_samples, self.dpi = every_n_epochs, n_samples, dpi

    def on_validation_end(self, trainer, pl_module):
        ep = trainer.current_epoch
        if ep % self.every_n_epochs:      # визуализируем только раз в N эпох
            return

        # --- берём 1 батч из val-датасета
        (z0, ts), true = next(iter(pl_module.val_dataloader()))
        z0, ts, true = z0.to(pl_module.device), ts.to(pl_module.device), true.to(pl_module.device)

        with torch.no_grad():
            pred = pl_module.rollout(z0, ts[0]-ts[0, 0], tol=pl_module.hparams.tol)

        # --- 1) траектории (по координате 0)
        fig_traj, ax = plt.subplots(figsize=(5, 3), dpi=self.dpi)
        for k in range(min(self.n_samples, z0.shape[0])):
            ax.plot(true[k, :, 0, 0, 0].cpu(), label=f"true {k}")
            ax.plot(pred[k, :, 0, 0, 0].cpu(), "--")
        ax.set_title("Validation trajectories"); ax.set_xlabel("t")
        ax.legend(ncol=2, fontsize=6)

        # --- 2) |ΔE|
        E_true = pl_module.true_energy(true)
        E_pred = pl_module.true_energy(pred)
        fig_E, axE = plt.subplots(figsize=(5, 2), dpi=self.dpi)
        axE.plot((E_pred-E_true).abs().mean(0).cpu())
        axE.set_title("|ΔE|"); axE.set_xlabel("t")

        # --- логируем в W&B
        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {"val/traj": fig_to_img(fig_traj), "val/energy_err": fig_to_img(fig_E)},
                step=ep,
            )

        plt.close(fig_traj); plt.close(fig_E)
