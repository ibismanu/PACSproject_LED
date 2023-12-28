import tensorflow as tf
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

class NNParams():
    batch_size : int
    epochs : int
    validation_split : float
    callbacks : list
    
    def __init__(self, batch_size=32, epochs=1000, validation_split=0.2, callbacks=None):
        self.batch_size=batch_size
        self.epochs=epochs
        self.validation_split=validation_split
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks=[
                tfk.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True),
                tfk.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5),
            ]
