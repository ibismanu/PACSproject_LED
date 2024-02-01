import numpy as np
import os

from scripts.datagen.fitzhugnagumo import FitzhugNagumo
from scripts.datagen.vanderpol import VanDerPol


def create_dataset(
    dataset_name,
    num_samples,
    num_processes,
    model_name=None,
    solver_params=None,
    model_params=None,
    batch_size=None,
    generate=True,
    remove_samples=True,
):
    if generate:

        if model_name=='Fitzhug Nagumo':

            FN = FitzhugNagumo(
                solver_params,
                k=model_params.k,
                alpha=model_params.alpha,
                epsilon=model_params.epsilon,
                I=model_params.I,
                gamma=model_params.gamma,
                grid_size=model_params.grid_size,
            )

            FN.generate_dataset(num_samples=num_samples, num_processes=num_processes)

        if model_name == 'Van Der Pol':
            VDP = VanDerPol(solver_params,model_params)

            VDP.generate_dataset(num_samples=num_samples,num_processes=num_processes)

    if batch_size is None:
        batch_size = num_samples
    num_batches = int(num_samples / batch_size) + 1

    dir_name = "../../dataset/" + dataset_name

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for b in range(num_batches - 1):
        merged = []
        for i in range(batch_size):
            filename = "../../dataset/samples/sample_" + str(b * batch_size + i) + ".npy"
            sample = np.load(filename)
            merged.append(sample)
        merged = np.array(merged)

        np.savez_compressed(
            dir_name + "/" + dataset_name + "_" + str(b),
            data=merged,
        )

    # merged = []
    # for i in range(num_samples % batch_size):
    #     filename = (
    #         "dataset/samples/sample_" + str((num_batches - 1) * (batch_size-1) + i) + ".npy"
    #     )
    #     print(filename)
    #     sample = np.load(filename)
    #     merged.append(sample)
    # merged = np.array(merged)

    # file_dir = dir_name + "/" + dataset_name + "_" + str(num_batches - 1) + ".npz"

    # np.savez_compressed(
    #     file=file_dir,
    #     data=merged,
    # )

    if remove_samples:
        for i in range(num_samples):
            filename = "../../dataset/samples/sample_" + str(i) + ".npy"
            os.remove(filename)
