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

    def get_at_vec(self, t_idx):
        xt = self.curr_x_traj[t_idx][:2]
        x_traj = self.curr_x_traj[:,:2]
        
        dfk_xt_all = np.array([
            -np.pi * self.ks[:,0] / self.L_list[0] * np.sin(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.cos(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
            -np.pi * self.ks[:,1] / self.L_list[1] * np.cos(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.sin(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
        ]) / self.hk_list

        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)

        at = np.sum(self.lamk_list * 2.0 * (ck_all - self.phik_list) * dfk_xt_all / (self.tsteps * self.dt), axis=1)
        return at

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return self.R @ ut 
    
    def loss(self, x_traj, u_traj):
        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)
        erg_metric = np.sum(self.lamk_list * np.square(ck_all - self.phik_list))

        ctrl_cost = np.sum(self.R @ u_traj.T * u_traj.T) * self.dt 
        return erg_metric + ctrl_cost 