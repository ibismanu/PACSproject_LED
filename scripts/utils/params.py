import tensorflow as tf
import numpy as np
import ast
tfk = tf.keras
tfkl = tfk.layers

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

    def get_from_file(self,filedir):
        
        with open(filedir, 'r') as file:
            lines = file.readlines()
        values = []

        for l in lines:
            values.append(l.strip().split(sep='='))

            if l[0]=='T':
                self.final_time = float(values[-1][1])
            if l[0:2]=='dt':
                self.time_step = float(values[-1][1])
            if l[0:2]=='u0':
                self.u0 = ast.literal_eval(values[-1][1])
        
        if 'ThetaMethod' in lines[3]:
            valid =  (lines[4][:5] == 'theta')
            assert valid, "the value theta is missing, or the format is incorrect"
            valid =  (lines[5][:3] == 'tol')
            assert valid, "the tolerance is missing, or the format is incorrect"
            self.theta = float(values[4][1])
            self.tol = float(values[5][1])

        if 'RungeKutta' in lines[3]:
            valid = (lines[4][0] == 'A')
            assert valid, "the matrix A is missing, or the format is incorrect"
            valid = (lines[5][0] == 'b')
            assert valid, "the vector b is missing, or the format is incorrect"
            valid = (lines[6][0] == 'c')
            assert valid, "the vector b is missing, or the format is incorrect"

            str_numbers = lines[4][2:] 
            components = str_numbers.rstrip('\n').split(' ')
            components_ = [ast.literal_eval(num) for num in components]
            arr = np.array(components_, dtype=np.float32)
            self.RK_A = arr

            str_numbers = ast.literal_eval(lines[5][2:])
            arr = np.array(str_numbers, dtype=np.float32)
            self.RK_b = arr

            str_numbers = ast.literal_eval(lines[6][2:]) 
            arr = np.array(str_numbers, dtype=np.float32)
            self.RK_c = arr

        if 'AdamsBashforth' in lines[3]:
            valid = (lines[4][:5] == 'order')
            assert valid, "the order is missing, or the format is incorrect"

            self.multi_order = int(lines[4][6:])

        

class FNParams:
    def __init__(
        self, solver_params, k, alpha, epsilon, I, gamma, grid_size, solver_name
    ):
        self.solver_params = solver_params
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
        self.solver_name = solver_name
