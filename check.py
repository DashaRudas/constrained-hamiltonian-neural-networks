# debug_magtrap.py -----------------------------------------------------------
import torch, os
from biases.datasets import MagneticTrapDataset   # ваш класс
from biases.datasets import MagneticTrap         # ваш класс

# ❶ принудительно регенерируем и сохраняем свежую копию
ds = MagneticTrapDataset(
        mode="train",
        n_systems=8,
        chunk_len=50,
        regen=True,           # <-- важно
)
print("Shapes from dataset:", ds.Ts.shape, ds.Zs.shape)   # ожидаем (N,50) (N,50,2,3)

# ❷ берём 1-й элемент как он пойдёт в model.rollout
(z0, ts), full_traj = ds[0]
print("z0 shape:", z0.shape, "ts shape:", ts.shape)
assert z0.ndim == 2 and z0.shape == (2,3), "z0 must be (2,3)"
assert ts.ndim == 1,                         "ts must be (T,)"

print("✅ dataset OK")

# ❸ хотите проверить integrate прямо сейчас?
trap = MagneticTrap()
pred = trap.integrate(z0.unsqueeze(0).reshape(1,6),
                      ts, rtol=1e-6, atol=1e-6)          # (T,1,6)
print("integrate output:", pred.shape)
# ----------------------------------------------------------------------------
