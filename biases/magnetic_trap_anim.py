# magnetic_trap_anim.py  ───── PATCHED VERSION ──────────────────────
import numpy as np, torch, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                     # noqa: F401
import matplotlib.animation as animation
from biases.animation import Animation


def _as_pos_vel(qt: torch.Tensor) -> torch.Tensor:
    """
    Принимает любой из вариантов:
        (T,6)        – [x,y,z,vx,vy,vz]
        (B,T,6)
        (T,2,3)      – уже (x,v)
        (B,T,2,3)
    и возвращает (T,2,3) или (B,T,2,3).
    """
    if qt.ndim == 2 and qt.shape[1] == 6:              # (T,6)
        return qt.view(-1, 2, 3)
    if qt.ndim == 3 and qt.shape[-1] == 6:             # (B,T,6)
        return qt.view(*qt.shape[:-1], 2, 3)
    return qt                                           # уже нужная форма


class MagneticTrapAnimation(Animation):
    """Анимация частицы в магнитной ловушке."""

    def __init__(self, qt: torch.Tensor, trap, tail: int = 100,
                 show_vel: bool = True):
        qt = _as_pos_vel(qt)                       # <-- главное исправление
        if qt.ndim == 4:                           # берём только первую систему
            qt = qt[0]
        super().__init__(qt, trap)

        self.tail, self.show_vel = tail, show_vel
        lim = float(qt[..., 0].abs().max()) * 1.2
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim); self.ax.set_zlim(-lim, lim)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("z")

        (self.point,) = self.ax.plot([], [], [], "ro", ms=6, zorder=10)
        (self.trail,) = self.ax.plot([], [], [], "b-", lw=1)
        self.vel = None                              # создадим в .init()

    # ------------------------------------------------------------------
    def init(self):
        self.point.set_data([], []); self.point.set_3d_properties([])
        self.trail.set_data([], []); self.trail.set_3d_properties([])
        if self.show_vel and self.vel is not None:
            self.vel.remove()
        self.vel = self.ax.quiver(0, 0, 0, 0, 0, 0, length=0)
        return self.point, self.trail, self.vel

    # ------------------------------------------------------------------
    def update(self, k=0):
        pos = self.qt[k, 0].cpu().numpy()
        self.point.set_data(pos[0], pos[1]); self.point.set_3d_properties(pos[2])

        s = max(0, k - self.tail)
        xyz_tail = self.qt[s:k + 1, 0].cpu().numpy().T
        self.trail.set_data(xyz_tail[0], xyz_tail[1])
        self.trail.set_3d_properties(xyz_tail[2])

        if self.show_vel:
            self.vel.remove()
            vel = self.qt[k, 1].cpu().numpy()
            self.vel = self.ax.quiver(*pos, *vel, length=0.4,
                                      color="g", normalize=True)
            return self.point, self.trail, self.vel

        return self.point, self.trail


# ―― monkey-patch -------------------------------------------------------------
from biases.datasets import MagneticTrap
MagneticTrap.animator = lambda self, zts, **kw: MagneticTrapAnimation(zts, self, **kw)
MagneticTrap.animate  = lambda self, zts, fps=30, **kw: (
    animation.FuncAnimation(
        MagneticTrapAnimation(zts, self, **kw).fig,
        MagneticTrapAnimation(zts, self, **kw).update,
        frames=_as_pos_vel(zts).shape[0],
        init_func=MagneticTrapAnimation(zts, self, **kw).init,
        interval=1000 / fps, blit=True
    ).to_jshtml()
)
# ─────────────────────────────────────────────────────────────────────────────
