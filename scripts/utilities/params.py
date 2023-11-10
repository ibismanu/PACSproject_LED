import numpy as np
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from utilities import utils

class Params():

    T: float
    dt: float
    u0: np.array

    def __init__(self, T: float, dt: float, u0,
                 theta=None, tol=None,
                 RK_A=None, RK_b=None, RK_c=None,
                 multi_A=None, multi_b=None, multi_order=None):

        self.T = T
        self.dt = dt
        self.theta = theta
        self.tol = tol
        self.RK_A = RK_A
        self.RK_b = RK_b
        self.RK_c = RK_c
        self.multi_A = multi_A
        self.multi_b = multi_b
        self.multi_order = multi_order

        if np.isscalar(u0):
            self.u0 = np.array([u0])
        else:
            self.u0 = u0


class ODEParams(Params):
    def __init__(self, T: float, dt: float, u0, f=lambda *args:None, theta=None, tol=None,
                 RK_A=None, RK_b=None, RK_c=None,
                 multi_A=None, multi_b=None, multi_order=None):
        super().__init__(T, dt, u0, theta=theta, tol=tol,
                 RK_A=RK_A, RK_b=RK_b, RK_c=RK_c,
                 multi_A=multi_A, multi_b=multi_b, multi_order=multi_order)
        self.f = utils.to_numpy(f)


class PDEParams(Params):
    def __init__(self, T: float, dt: float, u0, mass_matrix: np.array, system_matrix: np.array, forcing_term: np.array):
        super().__init__(T, dt, u0)
        self.mass_matrix = mass_matrix
        self.system_matrix = system_matrix
        self.forcing_term = forcing_term


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