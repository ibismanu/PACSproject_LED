import numpy as np
import os

from scripts.datagen.mitchellschaeffer import MitchellSchaeffer
from scripts.utils.params import MSParams, SolverParams


def create_dataset(
    dataset_name,
    num_samples,
    num_processes,
    ms_params,
    batch_size=None,
    generate=True,
    remove_samples=True,
):
    if generate:
        MS = MitchellSchaeffer(
            eqtype=ms_params.eqtype,
            params=ms_params.solver_params,
            k=ms_params.k,
            alpha=ms_params.alpha,
            epsilon=ms_params.epsilon,
            I=ms_params.I,
            gamma=ms_params.gamma,
            grid_size=ms_params.grid_size,
            solver_name=ms_params.solver_name,
        )

        MS.generate_dataset(num_samples=num_samples, num_processes=num_processes)

    if batch_size is None:
        batch_size = num_samples
    num_batches = int(num_samples / batch_size) + 1

    dir_name = "data/dataset/" + dataset_name

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for b in range(num_batches - 1):
        merged = []
        for i in range(batch_size):
            filename = "data/samples/sample_" + str(b * batch_size + i) + ".npy"
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
            "data/samples/sample_" + str((num_batches - 1) * (batch_size-1) + i) + ".npy"
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
            filename = "data/samples/sample_" + str(i) + ".npy"
            os.remove(filename)
