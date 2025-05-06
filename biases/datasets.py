import torch
import numpy as np
import os
import networkx as nx
from torch.utils.data import Dataset
from oil.utils.utils import Named, export
from biases.systems.rigid_body import RigidBody
from biases.systems.chain_pendulum import ChainPendulum
from biases.utils import rel_err, FixedSeedAll

@export
class RigidBodyDataset(Dataset, metaclass=Named):
    space_dim = 2
    num_targets = 1

    def __init__(
        self,
        root_dir=None,
        body=ChainPendulum(3),
        n_systems=100,
        regen=False,
        chunk_len=5,
        angular_coords=False,
        seed=0,
        mode="train",
        n_subsample=None,
    ):
        super().__init__()
        with FixedSeedAll(seed):
            self.mode = mode
            root_dir = root_dir or os.path.expanduser(
                f"~/datasets/ODEDynamics/{self.__class__}/"
            )
            self.body = body
            filename = os.path.join(
                root_dir, f"trajectories_{body}_N{n_systems}_{mode}.pz"
            )
            if os.path.exists(filename) and not regen:
                ts, zs = torch.load(filename)
            else:
                ts, zs = self.generate_trajectory_data(n_systems)
                os.makedirs(root_dir, exist_ok=True)
                torch.save((ts, zs), filename)
            Ts, Zs = self.chunk_training_data(ts, zs, chunk_len)

            if n_subsample is not None:
                Ts, Zs = Ts[:n_subsample], Zs[:n_subsample]
            self.Ts, self.Zs = Ts.float(), Zs.float()
            self.seed = seed
            if angular_coords:
                N, T = self.Zs.shape[:2]
                flat_Zs = self.Zs.reshape(N * T, *self.Zs.shape[2:])
                self.Zs = self.body.global2bodyCoords(flat_Zs.double())
                print(rel_err(self.body.body2globalCoords(self.Zs), flat_Zs))
                self.Zs = self.Zs.reshape(N, T, *self.Zs.shape[1:]).float()

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def generate_trajectory_data(self, n_systems, bs=10000):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """

        def base_10_to_base(n, b):
            """Writes n (originally in base 10) in base `b` but reversed"""
            if n == 0:
                return '0'
            nums = []
            while n:
                n, r = divmod(n, b)
                nums.append(r)
            return list(nums)

        batch_sizes = base_10_to_base(n_systems, bs)
        n_gen = 0
        t_batches, z_batches = [], []
        for i, batch_size in enumerate(batch_sizes):
            if batch_size == 0:
                continue
            batch_size = batch_size * (bs**i)
            print(f"Generating {batch_size} more chunks")
            z0s = self.sample_system(batch_size)
            ts = torch.arange(
                0, self.body.integration_time, self.body.dt, device=z0s.device, dtype=z0s.dtype
            )
            new_zs = self.body.integrate(z0s, ts)
            t_batches.append(ts[None].repeat(batch_size, 1))
            z_batches.append(new_zs)
            n_gen += batch_size
            print(f"{n_gen} total trajectories generated for {self.mode}")
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        n_trajs, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        # Cut each trajectory into non-overlapping chunks
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        # From each trajectory, we choose a single chunk randomly
        chunk_idx = torch.randint(0, n_chunks, (n_trajs,), device=zs.device).long()
        chosen_ts = chunked_ts[chunk_idx, range(n_trajs)]
        chosen_zs = chunked_zs[chunk_idx, range(n_trajs)]
        return chosen_ts, chosen_zs

    def sample_system(self, N):
        """"""
        return self.body.sample_initial_conditions(N)


class CartPole(RigidBody):
    def __init__(self):
        self.body_graph = nx.Graph()
        # Masses (and length) can be ignored for now since we are
        # only using the connectivity of the graph to
        # calculate DPhi, the constraint matrix
        self.body_graph.add_node(0, m=1, pos_cnstr=1)  # (axis)
        self.body_graph.add_node(1, m=1)
        self.body_graph.add_edge(0, 1, l=1)


class CartpoleDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        regen=False,
        batch_size=100,
        chunk_len=5,
        time_limit=10,
        seed=0,
    ):
        super().__init__()
        self.body = CartPole()
        self.seed = seed

        root_dir = root_dir or os.path.expanduser(
            f"~/datasets/ODEDynamics/{self.__class__}/"
        )
        filename = os.path.join(
            root_dir, f"trajectories_N{batch_size}_T{time_limit}.pz"
        )
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        else:
            ts, zs = self.generate_trajectory_data(batch_size, time_limit)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs), filename)
        self.Ts, self.Zs = self.chunk_training_data(ts, zs, chunk_len)

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def _initialize_env(self, time_limit, seed):
        from dm_control import suite
        import types

        env = suite.load(
            domain_name="cartpole", task_name="balance", task_kwargs={"random": seed}
        )

        # set time limit a la https://github.com/deepmind/dm_control/blob/03bebdf9eea0cbab480aa4882adbc4184b850835/dm_control/rl/control.py#L76
        if time_limit == float("inf"):
            env._step_limit = float("inf")
        else:
            # possible soure of error in the future if our agent does action repeats
            env._step_limit = time_limit / (env._physics.timestep() * env._n_sub_steps)

        def initialize_episode(self, physics):
            # replace https://github.com/deepmind/dm_control/blob/03bebdf9eea0cbab480aa4882adbc4184b850835/dm_control/suite/cartpole.py#L182
            physics.named.data.qpos["slider"][0] = np.array([0.0])
            physics.named.data.qpos["hinge_1"][0] = np.array([3.1415926 / 2])
            physics.named.data.qvel["slider"][0] = np.array([0.0])
            physics.named.data.qvel["hinge_1"][0] = np.array([0.0])

            # we'll use the default for Cartpole for now
            nv = physics.model.nv
            # if self._swing_up:
            #    physics.named.data.qpos["slider"] = 0.01 * self.random.randn()
            #     physics.named.data.qpos["hinge_1"] = np.pi + 0.01 * self.random.randn()
            #     physics.named.data.qpos[2:] = 0.1 * self.random.randn(nv - 2)
            # else:
            #     physics.named.data.qpos["slider"] = self.random.uniform(-0.1, 0.1)
            #     physics.named.data.qpos[1:] = self.random.uniform(-0.034, 0.034, nv - 1)
            # physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

            # call function from super class which should be a dm_control.suite.base.Task
            # replaces `super(Balance, self).initialize_episode(physics)`
            superclass = type(env.task).mro()[1]
            superclass.initialize_episode(self, physics)
            return

        # make `initialize_episode` a bound method instead of just a static function
        # https://tryolabs.com/blog/2013/07/05/run-time-method-patching-python/
        env.task.initialize_episode = types.MethodType(initialize_episode, env.task)

        return env

    def generate_trajectory_data(self, batch_size, time_limit):
        cartesian_trajs = []
        generalized_trajs = []
        for i in range(batch_size):
            if i % 100 == 0:
                print(i)
            # new random seed each run based on `self.seed`
            env = self._initialize_env(time_limit, self.seed + i)
            cartesian_traj, generalized_traj = self._evolve(env)
            cartesian_trajs.append(cartesian_traj)
            generalized_trajs.append(generalized_traj)

        cartesian_trajs, generalized_trajs = map(
            np.stack, [cartesian_trajs, generalized_trajs]
        )
        cartesian_trajs, generalized_trajs = map(
            torch.from_numpy, [cartesian_trajs, generalized_trajs]
        )
        time = torch.linspace(
            0,
            int(env._physics.timestep() * env._n_sub_steps * env._step_limit),
            int(env._step_limit) + 1 - 1,
        )  # add one because of initial state from env.reset(), subtract one because we throw away the last state to chunk our data evenly
        time = time.view(1, -1).expand(batch_size, len(time))

        # ignore generalized_trajs for now
        return time[:, ::50], cartesian_trajs[:, ::50]

    def _get_com(self, env, body):
        # we only keep x and z
        # from IPython.core.debugger import set_trace; set_trace()
        return np.copy(env.physics.named.data.xipos[body][[0, 2]])

    def _get_cvel(self, env, body):
        # last three entries corresponding to translational velocity
        # we only keep x and z
        return np.copy(env.physics.named.data.cvel[body][[-3, -1]])

    def _get_com_mom(self, env, body):
        return self._get_cvel(env, body) * env.physics.named.model.body_mass[body]

    def _get_q(self, env, body):
        return np.copy(env.physics.named.data.qpos[body])

    def _get_p(self, env, body):
        return np.copy(env.physics.named.data.qvel[body])

    def _evolve(self, env):
        # get the cartesian coordinate of pole and cart and their velocities
        time_step = env.reset()
        # positions
        pole_com = [self._get_com(env, "pole_1")]
        cart_com = [self._get_com(env, "cart")]
        pole_q = [self._get_q(env, "hinge_1")]
        cart_q = [self._get_q(env, "slider")]
        # momentums
        pole_com_mom = [self._get_com_mom(env, "pole_1")]
        cart_com_mom = [self._get_com_mom(env, "cart")]
        pole_p = [self._get_p(env, "hinge_1")]
        cart_p = [self._get_p(env, "slider")]

        action_spec = env.action_spec()
        while not time_step.last():
            # Take no action, just let the system evolve
            action = np.zeros(action_spec.shape)
            time_step = env.step(action)

            pole_com.append(self._get_com(env, "pole_1"))
            cart_com.append(self._get_com(env, "cart"))
            pole_q.append(self._get_q(env, "hinge_1"))
            cart_q.append(self._get_q(env, "slider"))

            pole_com_mom.append(self._get_com_mom(env, "pole_1"))
            cart_com_mom.append(self._get_com_mom(env, "cart"))
            pole_p.append(self._get_p(env, "hinge_1"))
            cart_p.append(self._get_p(env, "slider"))

        pole_com, pole_com_mom, cart_com, cart_com_mom = map(
            np.stack, [pole_com, pole_com_mom, cart_com, cart_com_mom]
        )
        pole_q, pole_p, cart_q, cart_p = map(np.stack, [pole_q, pole_p, cart_q, cart_p])

        com = np.stack([cart_com, pole_com])
        com_mom = np.stack([cart_com_mom, pole_com_mom])

        q = np.stack([cart_q, pole_q])
        p = np.stack([cart_p, pole_p])

        cartesian_traj = np.stack([com, com_mom])
        generalized_traj = np.stack([q, p])
        cartesian_traj = np.transpose(cartesian_traj, (2, 0, 1, 3))
        generalized_traj = np.transpose(generalized_traj, (2, 0, 1, 3))
        # output should be (number of time steps) x (q or p) x (number of bodys) x (dimension of quantity)

        # toss the final time step because we want to be able to chunk evenly
        cartesian_traj = cartesian_traj[:-1]
        generalized_traj = generalized_traj[:-1]
        return cartesian_traj, generalized_traj

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        batch_size, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = torch.randint(0, n_chunks, (batch_size,), device=zs.device).long()
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        chosen_ts = chunked_ts[chunk_idx, range(batch_size)]
        chosen_zs = chunked_zs[chunk_idx, torch.arange(batch_size).long()]
        return chosen_ts, chosen_zs


from torch import Tensor
from torchdiffeq import odeint                     # pip install torchdiffeq

class MagneticTrap:                                # аналог ChainPendulum, Gyroscope
    def __init__(self,
                 q: float = 1.0,                   # заряд
                 m: float = 1.0,                   # масса
                 dt: float = 1e-2,
                 integration_time: float = 10.0):
        self.D = 3              # q = (x,y,z)
        self.angular_dims = []  # все координаты линейные
        self.d = 3 
        self.q, self.m = q, m
        self.dt, self.integration_time = dt, integration_time
        import networkx as nx
        G = nx.Graph()
        G.add_node(0, m=self.m)
        self.body_graph = G

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
    def integrate(self, z0, ts,
                  tol=None,        # ← добавляем
                  rtol=1e-7, atol=1e-7,
                  method="dopri5"):

        # если пришёл старый аргумент tol – переиспользуем
        if tol is not None:
            rtol = atol = tol
        return odeint(self.f, z0, ts, rtol=rtol, atol=atol, method=method)

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
    """Dataset для частицы в магнитной ловушке — формат (N,T,2,3)."""

    def __init__(self,
                 root_dir="~/datasets/ODEDynamics/MagneticTrapDataset/",
                 n_systems=1000,
                 n_subsample=None,
                 trap=None, body=None,
                 chunk_len: int = 50,
                 regen: bool = False,
                 seed: int = 0,
                 mode: str = "train",
                 **_):
        
        super().__init__()
        self.trap = trap or body or MagneticTrap()
        torch.manual_seed(seed)
        self.body = self.trap 
        root_dir = os.path.expanduser(root_dir)
        os.makedirs(root_dir, exist_ok=True)
        fname = os.path.join(root_dir, f"traj_{mode}_N{n_systems}.pt")

        # --- читаем или генерируем -------------------------------------------------
        if regen or not os.path.exists(fname):
            ts, zs = self._gen_trajs(n_systems)
            torch.save((ts, zs), fname)
        else:
            ts, zs = torch.load(fname)

            # авто-конвертация старого формата (N,T,6) → (N,T,2,3)
            if zs.ndim == 3 and zs.shape[-1] == 6:
                x, v = zs[..., :3], zs[..., 3:]
                zs   = torch.stack([x, v], dim=2)        # (N,T,2,3)
                torch.save((ts, zs), fname)

        assert zs.ndim == 4 and zs.shape[-2:] == (2, 3), \
            f"Dataset wrong shape {zs.shape}, должны быть (N,T,2,3)"

        # --- нарезаем куски --------------------------------------------------------
        self.Ts, self.Zs = self._chunk(ts, zs, chunk_len)
        if n_subsample is not None:
            self.Ts, self.Zs = self.Ts[:n_subsample], self.Zs[:n_subsample]

    # -------------------------------------------------------------------------
    def _gen_trajs(self, N: int):
        z0 = self.trap.sample_initial_conditions(N)          # (N,6)
        ts = torch.arange(0., self.trap.integration_time, self.trap.dt)
        zs = self.trap.integrate(z0, ts).transpose(0, 1)     # (N,T,6)

        x, v = zs[..., :3], zs[..., 3:]
        zs   = torch.stack([x, v], dim=2)                    # (N,T,2,3)

        ts = ts.unsqueeze(0).repeat(N, 1)
        return ts, zs

    def _chunk(self, ts, zs, L):
        N, T = ts.shape
        n_chunks = T // L
        idx = torch.randint(0, n_chunks, (N,))
        return (ts.view(N, n_chunks, L)[torch.arange(N), idx].float(),
                zs.view(N, n_chunks, L, 2, 3)[torch.arange(N), idx].float())

    # dataset API -------------------------------------------------------------
    def __len__(self):      return len(self.Zs)
    def __getitem__(self,i): return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

