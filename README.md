# ML_customized_library_

🚀 Mini-ML Library & Autograd Engine (From Scratch)

This project is a complete implementation of a custom machine learning library built entirely from scratch using NumPy, without relying on frameworks like scikit-learn or PyTorch.
The goal is to deeply understand how machine learning models, neural networks, and automatic differentiation work internally by implementing them manually.

📌 Project Overview

This repository contains a reusable Python package my_ml_lib that mimics the functionality of mini versions of:

🔹 Scikit-learn (classical ML models & preprocessing)

🔹 PyTorch (autograd engine & neural networks)

The project includes implementations of classical machine learning algorithms, preprocessing tools, a custom autograd engine, and neural network training pipelines.

🧠 Key Features
🔹 Custom Machine Learning Library

Structured Python package with modular design:

Dataset loaders (Spambase, Fashion-MNIST)

Preprocessing tools (StandardScaler, Polynomial, Gaussian features)

Model selection utilities (train/val/test split, K-Fold)

Logistic Regression from scratch

One-vs-Rest multiclass classification

🔹 Autograd Engine (Mini-PyTorch)

Built a full automatic differentiation engine similar to PyTorch.

Supports:

Computation graph construction

Forward pass tracking

Backpropagation

Gradient computation

Implemented operations:

Arithmetic: add, sub, mul, div, pow

Matrix multiplication

Activations: ReLU, exp, log

Reductions: sum, mean

🔹 Neural Network Framework

Implemented deep learning components from scratch:

Linear layer

ReLU & Sigmoid activations

Sequential model container

CrossEntropy loss

SGD optimizer

Used to train Softmax regression and MLP models.

🔹 Experiments Performed

Models implemented and evaluated on real datasets:

OvR Logistic Regression

Softmax Regression (raw pixels)

Softmax + Polynomial Features

Softmax + Gaussian Features

MLP Neural Network

Includes:

Hyperparameter tuning

Validation-based model selection

Final test evaluation

Training loss visualization

Best model saving

🎯 Objective

To build a complete end-to-end machine learning and deep learning system from scratch and understand the internal working of:

Gradient descent

Backpropagation

Regularization

Feature scaling

Neural network training

Model generalization

🛠 Tech Stack

Python

NumPy

Matplotlib

No ML frameworks used

📊 Learning Outcome

This project demonstrates a deep understanding of machine learning fundamentals by recreating core ML and deep learning functionality from scratch, similar to building a mini version of scikit-learn + PyTorch.

👨‍💻 Author

Umesh Nehete
M.Tech AI — IIT Hyderabad

⭐ If you like this project

Give it a star ⭐ on GitHub!

Give it a star ⭐ on GitHub!
