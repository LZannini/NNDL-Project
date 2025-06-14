# Instructions

## Project Description

Consider the raw images from the MNIST dataset as input. This is a classification problem with C classes, where C= 10. Extract a global dataset of N pairs, and divide it appropriately into training and test sets (consider at least 10,000 elements for the training set and 2,500 for the test set). Use standard gradient descent as the weight update algorithm. Study the learning process of a neural network (e.g., epochs required for learning, error trend on training and validation sets, accuracy on the test set) with a single layer of internal nodes, varying the learning modality: online, batch, and mini-batch. Conduct this study for at least three different dimensions (number of nodes) for the internal layer. Select and keep activation functions constant. If necessary, due to computational time and memory constraints, you can reduce the dimensions of the raw MNIST dataset images (e.g., using the imresize function in MATLAB).

## Project Structure

```
nn_mnist/
│
├── model.py        *Defines the neural network*
├── train_eval.py   *Training and evaluation functions*
├── main.py         *Runs the experiments and plots results*
```

## Requirements

- Python 3.7+
- pip

## Required Python packages:

- `torch`
- `torchvision`
- `matplotlib`

```
pip install torch torchvision matplotlib
```

## Run 
```
python main.py
```
