import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data
from model import create_model
from utils import plot_results, convert_to_classes

def main():
    # Load data
    dog_data, cat_data = load_data('data/dogData_w.mat', 'data/catData_w.mat')
    
    # Prepare the dataset
    x_train = np.concatenate((dog_data[:, :40], cat_data[:, :40]), axis=1)
    x_test = np.concatenate((dog_data[:, 40:80], cat_data[:, 40:80]), axis=1)
    y_train = np.concatenate((np.ones((40, 1)), np.zeros((40, 1))), axis=0).T
    y_test = np.concatenate((np.zeros((40, 1)), np.ones((40, 1))), axis=0).T

    # Create and train the model
    model = create_model(input_shape=x_train.shape[0])
    model.fit(x_train.T, y_train, epochs=100, batch_size=10, verbose=1)

    # Evaluate the model
    y_train_pred = model.predict(x_train.T)
    y_test_pred = model.predict(x_test.T)

    # Convert predictions to class labels
    classes_train = convert_to_classes(y_train_pred)
    classes_test = convert_to_classes(y_test_pred)

    # Plot results
    plot_results(y_train_pred, y_test_pred)

if __name__ == "__main__":
    main()