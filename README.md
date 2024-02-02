# PACS project - LED

This project is an implementation of the Learning Effective Mechanism (LED) for the course Advanced Programming for Scientific Computing at Politecnico di Milano.

In the context of multiscale systems, it is crucial to effectively capture
the system’s dynamics and generate reliable predictions. In this framework, this
report presents the code structure supporting the implementation of the Learning
Effective Dynamics (LED) mechanism. The LED relies on surrogate models, em-
ploying an autoencoder and a Recurrent Neural Network. The code outlined in
this report guides the user through the dataset generation, advances to the con-
struction and training of neural networks, and culminates in the reproduction and
evaluation of the LED. The entire mechanism is tested and evaluated through two
case studies: the Van Der Pol oscillator and the Fitzhug-Nagumo model.

## Installation

The code requires [python 3.10](https://www.python.org/downloads/release/python-31013/) to run. It is run using the following libraries:
* [numpy](https://numpy.org/) v1.26.2
* [matplotlib](https://matplotlib.org/) v3.8.2
* [scipy](https://scipy.org/) v1.11.4
* [tensorflow](https://www.tensorflow.org/) v2.15.0
* [tqdm](https://pypi.org/project/tqdm/) v4.66.1


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install -r requirements.txt
```

## Usage
To run the code and reproduce the results, navigate the examples folder. Inside you can find 3 sub-folders.
In each sub-folder you find some Jupyter notebooks, each one referred to the specific example of the folder. To
well understand the functioning of the code, please read before the entire report.
The notebookes are the following:
* `dataset_creation`\
This file is designed to guide the reproduction of dataset creation. It provides a step-by-step explanation of the procedure for generating a new dataset.\
You can either run the notebook as it is or customize it according to your preferences.
Since an ordinary differential equation will be solved, a solver object from the class `GenerateParticle` is instantiated.\
You can change the parameters of the solver by creating a new parameter file (refer to the README file
inside `scripts/utils/parameters_files`) or by directly passing them to the constructor.\
Moreover, you can specify the dataset’s name and parameters for the specific problem. Through the call
to the function create_dataset (contained in `scripts/datagen`) you can define the number of samples
to generate and the number of processors for the parallel execution. The input value batch_size allows
you to group samples, while the boolean variable remove_samples enables the elimination of samples after
saving them in a single file. Adjusting batch_size is particularly useful for handling resource-intensive data generation, as in the case of the Fitzhug-Nagumo model.\
The notebook concludes by visualizing the generated dataset.
* `main`
By running this notebook, you can replicate the results concerning the LED network. The file specifies
3
the name of the pre-trained autoencoder and recurrent neural network (RNN), and employs the LED
class to process the data and reproduce the LED mechanism.\
The user has just to run the entire notebook, or to change the name of the dataset, of the autoencoder
or of the rnn to be uploaded.\
First, the object LED is initialized through the names of the pre-trained autoencoder and rnn, and the
desired length in time of the prediction. Then, the method run allows to run the network.. Finally, the method compute_error is called to compute the error
of the network using the norm in the specified order (L2 norm by default).
* `test_autoencoder`
By running this notebook, you can test separately the autoencoder used in the LED contest (i.e. used in
the main file). You just need to run the entire notebook.\
Through the creation of the `Autoencoder` object, by passing the model name to the constructor, the
pre-trained autoencoder will be automatically uploaded.
* `test_rnn`
As the previous function, the aim of this notebook is to test separately the rnn used in the main file for the LED network.
* `train_autoencoder`
You can either run this file without modifying anything, or customize it.\
The purpose of this notebook is to give the possibility to the user to build and train his own autoencoder.\
Once the object `Autoencoder` is initialized with the specified parameters (or the default ones), the
notebook calls the method get_data to prepare the dataset for the training. Moreover, the methods
build_model and train_model are called to respectively build and train the autoencoder accordingly to
the specified parameters.
* `train_rnn`
As the previous function, the objective of this notebook is to build and train a recurrent neural network.
You can either run this file without modifying anything, or customize it.

We underline that each notebook can be run as it is, without modifying anything.

If the user wants to create a new example, he has to create a new sub-folder inside the folder examples.
If he wants to create a dataset for the new example, it is necessary to build a new class that inherits from `DataGen`, since the generation of the dataset is problem-specific.
Furthermore, a new if instance with the name of the model has to be added inside the create_dataset
function.
The folder `examples/particles` contains also the Jupyter notebook `particles`. The purpose of this notebook is to test separately the solvers implemented inside `scripts/particle`.
