class SolverParams:
    def __init__(
        self,
        final_time,
        time_step,
        u0,
        theta=None,
        tol=None,
        RK_A=None,
        RK_b=None,
        RK_c=None,
        multi_A=None,
        multi_b=None,
        multi_order=None,
    ):
        self.final_time = final_time
        self.time_step = time_step
        self.u0 = u0
        self.theta = theta
        self.tol = tol
        self.RK_A = RK_A
        self.RK_b = RK_b
        self.RK_c = RK_c
        self.multi_A = multi_A
        self.multi_b = multi_b
        self.multi_order = multi_order


class MSParams:
    def __init__(
        self, eqtype, solver_params, k, alpha, epsilon, I, gamma, grid_size, solver_name
    ):
        self.eqtype = eqtype
        self.solver_params = solver_params
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
        self.solver_name = solver_name
