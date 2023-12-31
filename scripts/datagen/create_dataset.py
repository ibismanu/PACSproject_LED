import numpy as np
import os

from scripts.datagen.fitzhugnagumo import FitzhugNagumo
from scripts.utils.params import SolverParams, FNParams


def create_dataset(
    dataset_name,
    num_samples,
    num_processes,
    fn_params,
    batch_size=None,
    generate=True,
    remove_samples=True,
):
    if generate:
        FN = FitzhugNagumo(
            params=fn_params.solver_params,
            k=fn_params.k,
            alpha=fn_params.alpha,
            epsilon=fn_params.epsilon,
            I=fn_params.I,
            gamma=fn_params.gamma,
            grid_size=fn_params.grid_size,
            solver_name=fn_params.solver_name,
        )

        FN.generate_dataset(num_samples=num_samples, num_processes=num_processes)

    if batch_size is None:
        batch_size = num_samples
    num_batches = int(num_samples / batch_size) + 1

    dir_name = "dataset/" + dataset_name

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for b in range(num_batches - 1):
        merged = []
        for i in range(batch_size):
            filename = "dataset/samples/sample_" + str(b * batch_size + i) + ".npy"
            sample = np.load(filename)
            merged.append(sample)
        merged = np.array(merged)

        np.savez_compressed(
            dir_name + "/" + dataset_name + "_" + str(b),
            data=merged,
        )

    merged = []
    for i in range(num_samples % batch_size):
        filename = (
            "dataset/samples/sample_" + str((num_batches - 1) * (batch_size-1) + i) + ".npy"
        )
        print(filename)
        sample = np.load(filename)
        merged.append(sample)
    merged = np.array(merged)

    np.savez_compressed(
        dir_name
        + "/"
        + dataset_name
        + "_"
        + str(num_batches - 1),
        data=merged,
    )

    if remove_samples:
        for i in range(num_samples):
            filename = "dataset/samples/sample_" + str(i) + ".npy"
            os.remove(filename)
