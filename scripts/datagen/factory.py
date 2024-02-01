import numpy as np
import os

from scripts.datagen.fitzhugnagumo import FitzhugNagumo
from scripts.datagen.vanderpol import VanDerPol


def Model_Factory(model_name, solver_params, model_params):
    
    if model_name=='Fitzhug Nagumo':

        return FitzhugNagumo(
            solver_params,
            k=model_params.k,
            alpha=model_params.alpha,
            epsilon=model_params.epsilon,
            I=model_params.I,
            gamma=model_params.gamma,
            grid_size=model_params.grid_size,
        )
        
    if model_name == 'Van Der Pol':
        
        return VanDerPol(solver_params,model_params)