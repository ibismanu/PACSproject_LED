import tensorflow as tf
import numpy as np
import ast
tfk = tf.keras
tfkl = tfk.layers


# Class used to compact the parameters for the different types of solvers
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
        solver_name=None
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
        self.solver_name = solver_name

    @classmethod
    def get_from_file(cls,filedir):

        with open(filedir, 'r') as file:
            lines = file.readlines()
        values = []

        for l in lines:
            values.append(l.strip().split(sep='='))

            if l[0]=='T':
                final_time = float(values[-1][1])
            if l[0:2]=='dt':
                time_step = float(values[-1][1])
            if l[0:2]=='u0':
                u0 = ast.literal_eval(values[-1][1])

        if 'RK' in lines[3]:
            solver_name = 'rungekutta'
            
            return cls(final_time, time_step, u0, solver_name=solver_name)

        if 'ThetaMethod' in lines[3]:
            solver_name='thetamethod'
            valid =  (lines[4][:5] == 'theta')
            assert valid, "the value theta is missing, or the format is incorrect"
            valid =  (lines[5][:3] == 'tol')
            assert valid, "the tolerance is missing, or the format is incorrect"
            theta = float(values[4][1])
            tol = float(values[5][1])
            
            return cls(final_time, time_step, u0, theta=theta, tol=tol, solver_name=solver_name)

        if 'RungeKutta' in lines[3]:
            solver_name='rungekutta'
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
            RK_A = arr

            str_numbers = ast.literal_eval(lines[5][2:])
            arr = np.array(str_numbers, dtype=np.float32)
            RK_b = arr

            str_numbers = ast.literal_eval(lines[6][2:]) 
            arr = np.array(str_numbers, dtype=np.float32)
            RK_c = arr
            
            return cls(final_time, time_step, u0, RK_A=RK_A, RK_b=RK_b, RK_c=RK_c, solver_name=solver_name)
        
        if 'AdamsBashforth' in lines[3]:
            solver_name='multistep'
            valid = (lines[4][:5] == 'order')
            assert valid, "the order is missing, or the format is incorrect"
            multi_order = int(lines[4][6:])
            
            return cls(final_time, time_step, u0, multi_order=multi_order, solver_name=solver_name)

    def print_params(self):

        print("Solver = ", self.solver_name)
        print("T = ",self.final_time)
        print("dt = ", self.time_step)
        print('u0 = ', self.u0)

        if self.theta is not None:
            print("Theta: ", self.theta)
            print("Tolerance: ", self.tol)
        if self.multi_A is not None:
            print("A = ", self.RK_A)
            print("b = ", self.RK_b)
            print("c = ", self.RK_c)
        if self.multi_order is not None:
            print("A = ", self.multi_A)
            print("b = ", self.multi_b)
            print("Order = ", self.multi_order)
        

        

class FNParams:
    def __init__(
        self, k, alpha, epsilon, I, gamma, grid_size
    ):
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.I = I
        self.gamma = gamma
        self.grid_size = grid_size
