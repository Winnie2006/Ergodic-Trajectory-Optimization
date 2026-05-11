import numpy as np
from src.optimizer.ilqr_base import iLQR_template

class iLQR_ergodic_pointmass(iLQR_template):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
                 R, ks, L_list, lamk_list, hk_list, phik_list) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v)
        
        self.R = R 
        self.ks = ks 
        self.L_list = L_list
        self.lamk_list = lamk_list 
        self.hk_list = hk_list 
        self.phik_list = phik_list 

    def dyn(self, xt, ut):
        return ut
        
    def get_At_mat(self, t_idx):
        A = np.zeros((self.x_dim, self.x_dim))
        return A
    
    def get_Bt_mat(self, t_idx):
        B = np.eye(self.u_dim)
        return B

    # def get_at_vec(self, t_idx):
    #     xt = self.curr_x_traj[t_idx][:2]
    #     x_traj = self.curr_x_traj[:,:2]
        
    #     dfk_xt_all = np.array([
    #         -np.pi * self.ks[:,0] / self.L_list[0] * np.sin(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.cos(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
    #         -np.pi * self.ks[:,1] / self.L_list[1] * np.cos(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.sin(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
    #     ]) / self.hk_list

    #     fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
    #     ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)

    #     at = np.sum(self.lamk_list * 2.0 * (ck_all - self.phik_list) * dfk_xt_all / (self.tsteps * self.dt), axis=1)
    #     return at

    def get_at_vec(self, t_idx):

        # ==========================================
        # multi-agent reshape
        # ==========================================
        num_robot = self.x_dim // 2

        x_agents = self.curr_x_traj.reshape(
            self.tsteps,
            num_robot,
            2
        )

        xt_agents = x_agents[t_idx]

        # ==========================================
        # compute fk for all robots
        #
        # fk_all:
        # (T, N, K)
        # ==========================================
        fk_all = (
            np.prod(
                np.cos(
                    np.pi
                    * self.ks[None, None, :, :]
                    / self.L_list
                    * x_agents[:, :, None, :]
                ),
                axis=3
            )
            / self.hk_list[None, None, :]
        )

        # ==========================================
        # trajectory coefficients
        #
        # (N,K)
        # ==========================================
        ck_per_robot = (
            np.sum(fk_all, axis=0)
            * self.dt
            / (self.tsteps * self.dt)
        )

        # ==========================================
        # team-average coefficients
        #
        # (K,)
        # ==========================================
        ck_all = np.mean(ck_per_robot, axis=0)

        # ==========================================
        # compute dFk/dx
        #
        # result:
        # (N,K,2)
        # ==========================================
        dfk = np.zeros((num_robot, self.ks.shape[0], 2))

        for i in range(num_robot):

            xt = xt_agents[i]

            dfk[i,:,0] = (
                -np.pi
                * self.ks[:,0]
                / self.L_list[0]
                * np.sin(
                    np.pi
                    * self.ks[:,0]
                    / self.L_list[0]
                    * xt[0]
                )
                * np.cos(
                    np.pi
                    * self.ks[:,1]
                    / self.L_list[1]
                    * xt[1]
                )
            )

            dfk[i,:,1] = (
                -np.pi
                * self.ks[:,1]
                / self.L_list[1]
                * np.cos(
                    np.pi
                    * self.ks[:,0]
                    / self.L_list[0]
                    * xt[0]
                )
                * np.sin(
                    np.pi
                    * self.ks[:,1]
                    / self.L_list[1]
                    * xt[1]
                )
            )

        dfk /= self.hk_list[None,:,None]

        # ==========================================
        # ergodic gradient
        #
        # at_agents:
        # (N,2)
        # ==========================================
        coeff = (
            self.lamk_list
            * 2.0
            * (ck_all - self.phik_list)
            / (self.tsteps * self.dt)
        )

        at_agents = np.sum(
            coeff[None,:,None] * dfk,
            axis=1
        )

        # ==========================================
        # flatten back to joint vector
        #
        # (2N,)
        # ==========================================
        at = at_agents.reshape(-1)

        return at

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return self.R @ ut 
    
    def loss(self, x_traj, u_traj):

        # ==========================================
        # reshape trajectory into multi-agent form
        #
        # x_traj:
        # (T, x_dim)
        #
        # ->
        #
        # x_agents:
        # (T, num_robot, 2)
        # ==========================================
        num_robot = self.x_dim // 2
        x_agents = x_traj.reshape(x_traj.shape[0], num_robot, 2)

        # ==========================================
        # compute Fourier basis values
        #
        # fk_all:
        # (T, num_robot, K)
        # ==========================================
        fk_all = (
            np.prod(
                np.cos(
                    np.pi
                    * self.ks[None, None, :, :]
                    / self.L_list
                    * x_agents[:, :, None, :]
                ),
                axis=3
            )
            / self.hk_list[None, None, :]
        )

        # ==========================================
        # compute trajectory coefficients
        #
        # ck_all:
        # (num_robot, K)
        # ==========================================
        ck_all = (
            np.sum(fk_all, axis=0)
            * self.dt
            / (self.tsteps * self.dt)
        )

        # ==========================================
        # team-average ergodic statistics
        #
        # shape:
        # (K,)
        # ==========================================
        ck_all = np.mean(ck_all, axis=0)

        # ==========================================
        # ergodic metric
        # ==========================================
        erg_metric = np.sum(
            self.lamk_list
            * np.square(ck_all - self.phik_list)
        )

        # ==========================================
        # control effort cost
        # ==========================================
        ctrl_cost = (
            np.sum(self.R @ u_traj.T * u_traj.T)
            * self.dt
        )

        return erg_metric + ctrl_cost