import numpy as np
from src.optimizer.ilqr_base import iLQR_template

class AL_iLQR_template(iLQR_template):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
        penalty_init=1.0,
    ):
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v,)

        # AL parameters
        self.rho = penalty_init
        self.mu = None

    def objective(self, x_traj, u_traj):
        raise NotImplementedError

    def inequality(self, x_traj, u_traj):
        raise NotImplementedError

    # objective gradients
    def get_at_vec_objective(self, t_idx):
        raise NotImplementedError

    def get_bt_vec_objective(self, t_idx):
        raise NotImplementedError

    # constraint gradients
    def get_at_vec_constraint(self, t_idx):
        raise NotImplementedError

    def get_bt_vec_constraint(self, t_idx):
        raise NotImplementedError


    # augmented lagrangian
    def augmented_lagrangian(self, x_traj, u_traj):
        obj = self.objective(x_traj, u_traj)
        g = self.inequality(x_traj, u_traj)
        penalty = np.maximum(
            0.0,
            self.mu + self.rho * g
        )
        al_term = (
            0.5 / self.rho
            * np.sum(
                penalty**2 - self.mu**2
            )
        )
        return obj + al_term


    # AL gradients
    def get_at_vec(self, t_idx):

        at_obj = self.get_at_vec_objective(t_idx)

        g = self.inequality(
            self.curr_x_traj,
            self.curr_u_traj
        )

        multiplier = np.maximum(
            0.0,
            self.mu + self.rho * g
        )

        at_constr = self.get_at_vec_constraint(t_idx)

        at = at_obj + np.sum(
            multiplier[:, None] * at_constr,
            axis=0
        )

        return at

    def get_bt_vec(self, t_idx):

        bt_obj = self.get_bt_vec_objective(t_idx)

        g = self.inequality(
            self.curr_x_traj,
            self.curr_u_traj
        )

        multiplier = np.maximum(
            0.0,
            self.mu + self.rho * g
        )

        bt_constr = self.get_bt_vec_constraint(t_idx)

        bt = bt_obj + np.sum(
            multiplier[:, None] * bt_constr,
            axis=0
        )

        return bt

    # ==========================================
    # multiplier update
    # ==========================================

    def update_multiplier(self):

        g = self.inequality(
            self.curr_x_traj,
            self.curr_u_traj
        )

        self.mu = np.maximum(
            0.0,
            self.mu + self.rho * g
        )

    # ==========================================
    # line search
    # ==========================================

    def line_search(
        self,
        x0,
        u_traj,
        v_traj,
        loss_old,
        alpha_init=1.0,
        beta=0.5,
        max_iter=10,
    ):

        alpha = alpha_init

        for _ in range(max_iter):

            u_new = u_traj + alpha * v_traj

            x_new = self.traj_sim(x0, u_new)

            loss_new = self.augmented_lagrangian(
                x_new,
                u_new
            )

            if loss_new < loss_old:
                return u_new, alpha, loss_new

            alpha *= beta

        return u_traj, 0.0, loss_old

    # ==========================================
    # solve
    # ==========================================

    def solve(
        self,
        x0,
        u_init,
        max_iter=50,
    ):

        u_traj = u_init.copy()

        x_traj = self.traj_sim(x0, u_traj)

        self.curr_x_traj = x_traj.copy()
        self.curr_u_traj = u_traj.copy()

        # initialize multipliers
        g0 = self.inequality(x_traj, u_traj)

        self.mu = np.zeros_like(g0)

        loss_list = []
        violation_list = []
        rho_list = []
        traj_list = []

        for i in range(max_iter):

            print(f"\nAL Iter {i}")

            # ==================================
            # current loss
            # ==================================
            loss_old = self.augmented_lagrangian(
                self.curr_x_traj,
                self.curr_u_traj
            )

            loss_list.append(loss_old)

            # ==================================
            # Riccati descent
            # ==================================
            v_traj = self.get_descent(
                x0,
                self.curr_u_traj
            )

            # ==================================
            # line search
            # ==================================
            u_new, alpha, loss_new = self.line_search(
                x0,
                self.curr_u_traj,
                v_traj,
                loss_old,
            )

            # ==================================
            # update trajectory
            # ==================================
            x_new = self.traj_sim(x0, u_new)

            self.curr_x_traj = x_new.copy()
            self.curr_u_traj = u_new.copy()

            # ==================================
            # update multipliers
            # ==================================
            self.update_multiplier()

            # ==================================
            # update penalty
            # ==================================
            g = self.inequality(
                self.curr_x_traj,
                self.curr_u_traj
            )

            violation = np.sum(
                np.maximum(0.0, g)
            )
            
            violation_list.append(violation)
            rho_list.append(self.rho)
            traj_list.append(self.curr_x_traj.copy())

            if violation > 1e-3:
                self.rho *= 1.05

            print(
                f"loss={loss_new:.6f}, "
                f"alpha={alpha:.4f}, "
                f"violation={violation:.6f}, "
                f"rho={self.rho:.4f}"
            )

        return (
            self.curr_u_traj,
            loss_list,
            violation_list,
            rho_list,
            traj_list,
        )