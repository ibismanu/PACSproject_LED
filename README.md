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

