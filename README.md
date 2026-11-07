# DigitLens-CNN-Based-Handwritten-Digit-Recognition
Based on your uploaded assignment, the project will use MNIST and combine both required parts into one application.



Part A — CNN Image Classification

Train a Convolutional Neural Network to recognize handwritten digits from 0 to 9, evaluate its performance, and visualize correct and incorrect predictions.
Part B — Streamlit Web Application

Build an interactive interface where users upload handwritten digit images, receive predictions, view confidence scores, and explore model performance.

Your assignment requires exactly two submission files: one Python file containing both parts and one PDF report, preferably 4–6 pages. The project also needs to run successfully and include evidence of training and evaluation.
Implementation details
The Python program now includes a CNN with two convolutional layers, pooling, dropout, and a 10-class output. It also supports a reproducible training/validation split, saved model weights, prediction charts, and invalid-image handling.

Local checks passed: CNN forward pass, backpropagation, image preprocessing, confidence-score calculation, and blank-image rejection. These are functional checks, not yet measured MNIST accuracy.
