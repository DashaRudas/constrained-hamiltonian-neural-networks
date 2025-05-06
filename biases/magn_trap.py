# ─────────────────────────────────────────────────────────────────────────────
# 1.  ФИЗИЧЕСКАЯ СИСТЕМА  ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
import torch
from torch import Tensor
from torchdiffeq import odeint                     # pip install torchdiffeq

class MagneticTrap:                                # аналог ChainPendulum, Gyroscope
    def __init__(self,
                 q: float = 1.0,                   # заряд
                 m: float = 1.0,                   # масса
                 dt: float = 1e-2,
                 integration_time: float = 10.0):
        self.q, self.m = q, m
        self.dt, self.integration_time = dt, integration_time

    # ---------- 1.1 поле  ---------------------------------------------------
    def B(self, x: Tensor) -> Tensor:
        """
        МЕСТО ДЛЯ ВАШЕЙ РЕАЛИЗАЦИИ.
        x : (..., 3)  →  B(x) : (..., 3)
        Можно вызвать свой solвер dB/ds или просто вернуть константу/градиент.
        """
        # пример квадратичной ловушки (магнитное бутылочное поле)
        # Bz = B0 + (G/2)*(x^2 + y^2 - 2 z^2);  Bx=By=0
        B0, G = 1.0, 0.1
        x2y2 = (x[...,0]**2 + x[...,1]**2)
        Bz   = B0 + 0.5*G*(x2y2 - 2*x[...,2]**2)
        return torch.stack([torch.zeros_like(Bz), torch.zeros_like(Bz), Bz], dim=-1)

    # ---------- 1.2 прав.-часть d/dt (x,v) ----------------------------------
    def f(self, t: Tensor, z: Tensor) -> Tensor:
        """
        z = (x, v)  сshape (..., 6)
        Возвращает d/dt (x,v) = (v,  (q/m)*v×B(x) )
        """
        x, v = z[..., :3], z[..., 3:]
        B = self.B(x)
        a = (self.q/self.m) * torch.cross(v, B, dim=-1)   # Лоренц a = (q/m) v×B
        return torch.cat([v, a], dim=-1)

    # ---------- 1.3 интегратор ---------------------------------------------
    def integrate(self, z0: Tensor, ts: Tensor, rtol=1e-7, atol=1e-7):
        # z0: (batch, 6); ts: (T,) или (batch,T)
        return odeint(self.f, z0, ts, rtol=rtol, atol=atol)  # (T,batch,6)

    # ---------- 1.4 генерация начальных условий ----------------------------
    def sample_initial_conditions(self, N: int) -> Tensor:
        """
        Равномерно в шаре r<1 и Maxwellian скорости; поправьте по желанию.
        Возвращает (N,6)  [x0_y0_z0_vx_vy_vz]
        """
        # позиции
        xyz = torch.randn(N,3)
        xyz = xyz / xyz.norm(dim=-1,keepdim=True) * torch.rand(N,1)**(1/3)
        # скорости
        v = torch.randn(N,3)
        return torch.cat([xyz, v], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET  ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset
import os

class MagneticTrapDataset(Dataset):
    """Аналог RigidBodyDataset, но для MagneticTrap"""
    def __init__(self,
                 root_dir="~/datasets/MagTrap/",
                 n_systems=1000,
                 n_subsample=None,
                 trap: MagneticTrap = MagneticTrap(),
                 chunk_len: int = 50,
                 regen: bool = False,
                 seed: int = 0,
                 mode: str = "train"):
        super().__init__()
        torch.manual_seed(seed)
        root_dir = os.path.expanduser(root_dir)
        os.makedirs(root_dir, exist_ok=True)

        fname = os.path.join(root_dir,
                             f"traj_{mode}_N{n_systems}.pt")
        if os.path.exists(fname) and not regen:
            ts, zs = torch.load(fname)
        else:
            ts, zs = self._gen_trajs(trap, n_systems)
            torch.save((ts, zs), fname)

        # — разбиваем длинные траектории на куски одинаковой длины
        self.Ts, self.Zs = self._chunk(ts, zs, chunk_len)

        if n_subsample is not None:
            self.Ts, self.Zs = self.Ts[:n_subsample], self.Zs[:n_subsample]

    # ---------- dataset API -------------------------------------------------
    def __len__(self):  return len(self.Zs)
    def __getitem__(self,i):
        return (self.Zs[i,0], self.Ts[i]), self.Zs[i]          # (z0, t_grid), full traj

    # ---------- helpers -----------------------------------------------------
    def _gen_trajs(self, trap: MagneticTrap, N: int):
        """ N систем × целая траектория """
        z0 = trap.sample_initial_conditions(N)                 # (N,6)
        ts = torch.arange(0., trap.integration_time, trap.dt)  # (T,)
        zs = trap.integrate(z0, ts).transpose(0,1)             # (N,T,6)
        ts = ts.unsqueeze(0).repeat(N,1)                       # (N,T)
        return ts, zs

    def _chunk(self, ts: Tensor, zs: Tensor, L: int):
        """рандомно вырезаем фрагменты длиной L"""
        N,T = ts.shape
        n_chunks = T // L
        idx = torch.randint(0, n_chunks, (N,))
        ts_chunks = ts.view(N,n_chunks,L)[torch.arange(N),idx]
        zs_chunks = zs.view(N,n_chunks,L,6)[torch.arange(N),idx]
        return ts_chunks.float(), zs_chunks.float()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ПРОВЕРКА  ---------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = MagneticTrapDataset(n_systems=4, chunk_len=20, regen=True)
    print("Dataset shape:", ds.Zs.shape)          # → (4,20,6)
    (z0, t), traj = ds[0]
    print("One chunk:", z0.shape, t.shape, traj.shape)
