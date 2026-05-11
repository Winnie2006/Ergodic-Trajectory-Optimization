import numpy as np
from src.optimizer.ilqr_base import iLQR_template
from src.dynamics.double_integrator import DoubleIntegrator
from src.ergodic.metric import generate_k_vectors, compute_phi


class ErgodicILQR(iLQR_template):
    def __init__(self, dt, tsteps, pdf, L_list, grid_size,
                 Q_z, R_v, num_k=10):

        self.dyn_sys = DoubleIntegrator(dim=2)

        super().__init__(
            dt=dt,
            tsteps=tsteps,
            x_dim=self.dyn_sys.nx,
            u_dim=self.dyn_sys.nu,
            Q_z=Q_z,
            R_v=R_v
        )

        # ergodic params
        self.L_list = L_list
        self.ks = generate_k_vectors(num_k)

        grids, dx, dy = self._create_grid(L_list, grid_size)
        self.phik = compute_phi(pdf, self.ks, grids, dx, dy, L_list)

    def _create_grid(self, L_list, grid_size):
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, L_list[0], grid_size),
            np.linspace(0, L_list[1], grid_size)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

        dx = L_list[0] / (grid_size - 1)
        dy = L_list[1] / (grid_size - 1)

        return grids, dx, dy

    # dynamics 
    def dyn(self, xt, ut):
        return np.array(self.dyn_sys.derivative(xt, ut))

    # linearization
    def get_At_mat(self, t_idx):
        xt = self.curr_x_traj[t_idx]
        ut = self.curr_u_traj[t_idx]
        return np.array(self.dyn_sys.getAt(xt, ut))

    def get_Bt_mat(self, t_idx):
        xt = self.curr_x_traj[t_idx]
        ut = self.curr_u_traj[t_idx]
        return np.array(self.dyn_sys.getBt(xt, ut))

    # ergodic gradient
    def _fk(self, x, k):
        return np.prod(np.cos(np.pi * k / self.L_list * x))

    def _dfk_dx(self, x, k):
        grad = np.zeros(2)
        for i in range(2):
            val = -np.pi * k[i] / self.L_list[i] * np.sin(
                np.pi * k[i] * x[i] / self.L_list[i]
            )
            for j in range(2):
                if j != i:
                    val *= np.cos(np.pi * k[j] * x[j] / self.L_list[j])
            grad[i] = val
        return grad

    def _compute_ck(self, x_traj):
        ck = np.zeros(len(self.ks))
        for t in range(self.tsteps):
            x = x_traj[t][:2]
            for i, k in enumerate(self.ks):
                ck[i] += self._fk(x, k)
        ck /= self.tsteps
        return ck

    # cost gradient
    def get_at_vec(self, t_idx):
        x = self.curr_x_traj[t_idx][:2]

        ck = self._compute_ck(self.curr_x_traj)

        grad = np.zeros(2)
        for i, k in enumerate(self.ks):
            dfk = self._dfk_dx(x, k)
            grad += 2 * (ck[i] - self.phik[i]) * dfk

        full_grad = np.zeros(self.x_dim)
        full_grad[:2] = grad
        return full_grad

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return 2 * self.R_v @ ut

    # loss: objective function 
    def loss(self):
        ck = self._compute_ck(self.curr_x_traj)

        # [CHECK] 
        print(f"[CHECK] ck mean: {np.mean(ck):.4e}, std: {np.std(ck):.4e}")
        print(f"[CHECK] phik mean: {np.mean(self.phik):.4e}")
        
        erg = np.sum((ck - self.phik)**2)
        ctrl = np.sum([u @ self.R_v @ u for u in self.curr_u_traj])
        return erg + ctrl