import jax.numpy as jnp


# from utils.integrator import euler as int_func
from utils.integrator import rk4 as int_func
from functools import partial
from jax.lax import scan
from jax import jit, vmap, grad


# see https://github.com/MurpheyLab/ergodic-control-sandbox/blob/main/notebooks/ilqr_ergodic_control.ipynb
# I speed up the algorithm
class iLQR_template:
    def __init__(self, dt, tsteps, Q_z, R_v, dynamics: callable) -> None:
        self.dt = dt
        self.tsteps = tsteps
        self.tf = dt * tsteps

        self.x_dim = getattr(dynamics, "Nx", getattr(dynamics, "nx", None))
        self.u_dim = getattr(dynamics, "Nu", getattr(dynamics, "nu", None))
        self.dynamics = dynamics

        self.Q_z = Q_z
        self.Q_z_inv = jnp.linalg.inv(Q_z)
        self.R_v = R_v
        self.R_v_inv = jnp.linalg.inv(R_v)

        self.dyn_step = jit(partial(int_func, dxdt=self.dynamics.dxdt, dt=self.dt))

        def dyn_step_fn(x, u):
            return self.dyn_step(xt=x, u=u), x

        self.dyn_step_fn = jit(dyn_step_fn)

        # the following functions are utilities for solving the Riccati equation
        # P
        def P_dyn_rev(Pt, At, Bt, at, bt):
            return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_v_inv @ Bt.T @ Pt + self.Q_z

        self.P_dyn_step = jit(partial(int_func, dxdt=P_dyn_rev, dt=self.dt))

        def P_dyn_step_fn(Pt, inputs):
            # def P_dyn_step_fn(Pt, At, Bt, at, bt):
            At, Bt, at, bt = inputs["At"], inputs["Bt"], inputs["at"], inputs["bt"]
            return self.P_dyn_step(xt=Pt, At=At, Bt=Bt, at=at, bt=bt), Pt

        self.P_dyn_step_fn = jit(P_dyn_step_fn)

        # r
        def r_dyn_rev(rt, Pt, At, Bt, at, bt):
            return (
                (At - Bt @ self.R_v_inv @ Bt.T @ Pt).T @ rt
                + at
                - Pt @ Bt @ self.R_v_inv @ bt
            )

        self.r_dyn_step = jit(partial(int_func, dxdt=r_dyn_rev, dt=self.dt))

        def r_dyn_step_fn(rt, inputs):
            Pt, At, Bt, at, bt = (
                inputs["Pt"],
                inputs["At"],
                inputs["Bt"],
                inputs["at"],
                inputs["bt"],
            )
            # def r_dyn_step_fn(rt, Pt, At, Bt, at, bt):
            return self.r_dyn_step(xt=rt, Pt=Pt, At=At, Bt=Bt, at=at, bt=bt), rt

        self.r_dyn_step_fn = jit(r_dyn_step_fn)

        # z /delta
        def z2v(zt, Pt, rt, Bt, bt):
            return (
                -self.R_v_inv @ Bt.T @ Pt @ zt
                - self.R_v_inv @ Bt.T @ rt
                - self.R_v_inv @ bt
            )

        self.z2v = jit(z2v)

        def z_dyn(zt, Pt, rt, At, Bt, bt):
            return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)

        self.z_dyn_step = jit(partial(int_func, dxdt=z_dyn, dt=self.dt))

        def z_dyn_step_fn(zt, inputs):
            # def z_dyn_step_fn(zt, Pt, rt, At, Bt, bt):
            Pt, rt, At, Bt, bt = (
                inputs["Pt"],
                inputs["rt"],
                inputs["At"],
                inputs["Bt"],
                inputs["bt"],
            )
            return self.z_dyn_step(xt=zt, Pt=Pt, rt=rt, At=At, Bt=Bt, bt=bt), zt

        self.z_dyn_step_fn = jit(z_dyn_step_fn)

        # self.temp = {'A_traj':[], 'B_traj':[], 'a_traj':[], 'b_traj':[], 'P_traj':[], 'r_traj':[], 'z_traj':[], 'v_traj':[], 'x_traj':[], 'u_traj':[]}

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_bt_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_bt_traj(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def traj_sim(self, x0, u_traj):
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        return x_traj

    def get_descent(self, x0, u_traj):
        # forward simulate the trajectory
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        # x_traj = jnp.vstack([x_traj, xN[jnp.newaxis, :]])[1:] ## the index is important
        self.curr_x_traj = x_traj.copy()
        self.curr_u_traj = u_traj.copy()

        # sovle the Riccati equation backward in time
        A_traj = vmap(self.dynamics.getAt)(x_traj, u_traj)
        B_traj = vmap(self.dynamics.getBt)(x_traj, u_traj)
        # t_idx = jnp.arange(self.tsteps)
        a_traj, b_traj = self.get_at_bt_traj({"x": x_traj, "u": u_traj})
        # try:
        #     a_traj, b_traj = self.get_at_bt_traj({"x": x_traj, "u": u_traj})
        # except (AttributeError, NotImplementedError):
        #     a_traj = vmap(self.get_at_vec, in_axes=(0, None))(t_idx, x_traj)
        #     b_traj = vmap(self.get_bt_vec, in_axes=(0, None))(t_idx, u_traj)

        # self.A_traj = A_traj.copy()
        # self.B_traj = B_traj.copy()
        # self.a_traj = a_traj.copy()
        # self.b_traj = b_traj.copy()

        PN = jnp.zeros((self.x_dim, self.x_dim))
        P0, P_traj = scan(
            f=self.P_dyn_step_fn,
            init=PN,
            reverse=True,
            xs={"At": A_traj, "Bt": B_traj, "at": a_traj, "bt": b_traj},
        )
        P_traj = jnp.vstack([P0[jnp.newaxis, :], P_traj])[:-1]
        # self.P_traj = P_traj.copy()

        rN = jnp.zeros(self.x_dim)
        r0, r_traj = scan(
            f=self.r_dyn_step_fn,
            init=rN,
            reverse=True,
            xs={"Pt": P_traj, "At": A_traj, "Bt": B_traj, "at": a_traj, "bt": b_traj},
        )
        r_traj = jnp.vstack([r0[jnp.newaxis, :], r_traj])[:-1]
        # self.r_traj = r_traj.copy()

        z0 = jnp.zeros(self.x_dim)
        zN, z_traj = scan(
            f=self.z_dyn_step_fn,
            init=z0,
            xs={"Pt": P_traj, "rt": r_traj, "At": A_traj, "Bt": B_traj, "bt": b_traj},
        )
        # z_traj = jnp.vstack([z_traj, zN[jnp.newaxis, :]])[1:] ## the index is important
        # self.z_traj = z_traj.copy()

        # compute the descent direction
        v_traj = vmap(self.z2v)(z_traj, P_traj, r_traj, B_traj, b_traj)

        return v_traj

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")


# argumented ilqr method
class al_iLQR(iLQR_template):
    def __init__(
        self, args: dict, objective: callable, dynamics: callable, inequality: callable
    ) -> None:
        super().__init__(
            dt=args["dt"],
            tsteps=args["tsteps"],
            Q_z=args["Q_z"],
            R_v=args["R_v"],
            dynamics=dynamics,
        )
        self.objective = jit(objective)
        self.inequality = jit(inequality)
        self.R = args["R"]

        # self.U_min = self.dynamics.U_min
        # self.U_max = self.dynamics.U_max
        self.U_min = args["U_min"]
        self.U_max = args["U_max"]

        self.r_penalty = 1.0
        self.dual_solution = None
        self.init_state = None
        self.solution = None

        def lagrangian(solution, dual_solution, r):
            # lam = dual_solution["lam"]
            mu = dual_solution["mu"]
            _objective = self.objective(solution)
            _ineq_constr = self.inequality(solution)

            return _objective + (0.5 / r) * jnp.sum(
                jnp.maximum(0.0, mu + r * _ineq_constr) ** 2 - mu**2
            )

        self.lagrangian = jit(lagrangian)
        self.lagrangian_grad = jit(grad(lagrangian, argnums=0))

        def _loss_func(_step, _u_current, _u_direct, dual, penalty):
            ctrl = jnp.clip(_u_current + _step * _u_direct, self.U_min, self.U_max)
            x_traj = self.traj_sim(self.init_state, ctrl)
            return self.lagrangian({"x": x_traj, "u": ctrl}, dual, penalty)

        self.loss_func4linesearch = jit(_loss_func)

    def get_at_bt_traj(self, solution):
        grad_val = self.lagrangian_grad(solution, self.dual_solution, self.r_penalty)
        return grad_val["x"], grad_val["u"]

    def update_multipliers(self, solution):
        # self.dual_solution["lam"] = self.dual_solution["lam"] + self.r_penalty * self.equality(solution)
        self.dual_solution["mu"] = jnp.maximum(
            0, self.dual_solution["mu"] + self.r_penalty * self.inequality(solution)
        )

    def linesearch(
        self, u_current, u_direct, max_iter=50, initial_step=1.0, beta=0.8, sigma=0.1
    ):
        steps_arr = jnp.array([initial_step * beta**i for i in range(max_iter)])
        loss_arr = vmap(
            jit(
                partial(
                    self.loss_func4linesearch,
                    _u_current=u_current,
                    _u_direct=u_direct,
                    dual=self.dual_solution,
                    penalty=self.r_penalty,
                )
            )
        )(_step=steps_arr)
        min_loss_idx = jnp.argmin(loss_arr)
        min_step = steps_arr[min_loss_idx]
        min_loss = loss_arr[min_loss_idx]
        if min_loss <= self.lagrangian(
            self.solution, self.dual_solution, self.r_penalty
        ):
            # if min_loss <= self.lagrangian(
            #     self.solution, self.dual_solution, self.r_penalty
            # ) + sigma * min_step * jnp.dot(u_direct.flatten(), u_direct.flatten()):
            ctrl = jnp.clip(u_current + min_step * u_direct, self.U_min, self.U_max)
            # print(min_step)
            return ctrl
        else:
            return u_current

    def solve(self, x0, init_sol: dict, max_iter=100, decay_eps=0.1, if_print=True):
        self.init_state = x0
        self.solution = init_sol
        # initialize the dual solution
        self.dual_solution = {"mu": jnp.zeros_like(self.inequality(self.solution))}
        self.update_multipliers(init_sol)

        loss_val = [self.lagrangian(self.solution, self.dual_solution, self.r_penalty)]
        _func_get_violation = jit(
            lambda sol: jnp.maximum(0, self.inequality(sol)).sum()
        )
        violations = [_func_get_violation(self.solution)]

        # iterative optimization
        for _ in range(max_iter):
            # solver LQR Problem
            v_traj = self.get_descent(self.init_state, self.solution["u"])
            # line search
            # _u_traj = self.linesear
            _u_traj = self.linesearch(u_current=self.solution["u"], u_direct=v_traj)
            # update solution
            self.solution["u"] = _u_traj
            # trajectory simulation
            self.solution["x"] = self.traj_sim(self.init_state, self.solution["u"])
            loss_val.append(
                self.lagrangian(self.solution, self.dual_solution, self.r_penalty)
            )
            violations.append(_func_get_violation(self.solution))
            if if_print:
                print(
                    "iter: {:d}\tobjective: {:.5f}\tlagrangian: {:.5f}\tviolation: {:.5f}\tpenalty: {:.5f}".format(
                        _,
                        self.objective(self.solution),
                        loss_val[-1],
                        violations[-1],
                        self.r_penalty,
                    )
                )
            # check if update multipliers
            if (loss_val[-2] - loss_val[-1]) > decay_eps:
                self.update_multipliers(self.solution)
            else:
                self.r_penalty = jnp.clip(self.r_penalty * 1.05, 1e-10, 1e10)
                decay_eps *= 0.95
            if (jnp.abs(loss_val[-1] - loss_val[-2]) < 1e-6) and jnp.abs(
                violations[-1]
            ) < 1e-2:
                print("iter", _, self.r_penalty)
                return self.solution

        if jnp.abs(violations[-1]) > 1e-2:
            print("failed to satisfy inequality, iter", _, self.r_penalty)
        else:
            print("satisfy inequality, but not converge, iter", _, self.r_penalty)
        return self.solution
