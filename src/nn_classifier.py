# path tools
import os
import sys

import pandas as pd

# image processing
import cv2

# neural networks with numpy
import numpy as np

# utils
from utils.neuralnetwork import NeuralNetwork

# neural networks with tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

import argparse

# a function that allows that user to choose which dataset they want to work with - either "mnist_784" ot "cifar10"
def choose(data):
    dataset = data
    return dataset

# if the user chooses "mnist_784", this function will run
def load_and_process_mnist(dataset):
    # load the data
    X, y = fetch_openml(dataset, return_X_y = True)
    # convert to arrays
    X = np.array(X)
    y = np.array(y)
    # create train-test-splot
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=9,
                                                    train_size=7500, 
                                                    test_size=2500)
    # define label binarizer
    lb = LabelBinarizer()
    #binarize labels
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    # normalize
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255
    return X_train_scaled, X_test_scaled, y_train, y_test

# if the user chooses "cifar10", this function will run
def load_and_process_cifar10():
    # load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # define the labels - STILL NOT SURE HOW TO IMPLEMENT THIS IN THE CLF-REPORT
    labels = ["airplane", 
          "automobile", 
          "bird",
          "cat",
          "deer",
          "dog",
          "frog",
          "horse",
          "ship",
          "truck"]
    # define label binarizer
    lb = LabelBinarizer()
    # binarize labels
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test) 
    # convert color images to grayscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # convert to arrays and normalize
    X_train_norm = np.array(X_train_grey/255)
    X_test_norm = np.array(X_test_grey/255)
    # reshape to 1-arrays
    nsamples, nx, ny = X_train_norm.shape
    X_train_scaled = X_train_norm.reshape((nsamples, nx*ny))
    nsamples, nx, ny = X_test_norm.shape
    X_test_scaled = X_test_norm.reshape((nsamples, nx*ny))
    return X_train_scaled, X_test_scaled, y_train, y_test

# a simple neural network classifier 
def nn_model(X_train_scaled, y_train):
    # define parameters
    input_shape = X_train_scaled.shape[1]
    # initialize neural network
    nn = NeuralNetwork([input_shape, 64, 10]) 
    # fit the data to the neural network
    nn.fit(X_train_scaled, y_train,  epochs = 10, displayUpdate=1) 
    return nn

# get predictions
def nn_predictor(nn, X_test_scaled, y_test):
    predictions = nn.predict(X_test_scaled)
    y_pred = predictions.argmax(axis=1)
    # print classification report without additional parameter
    print(classification_report(y_test.argmax(axis=1), y_pred))
    # get classification report
    report = classification_report(y_test.argmax(axis=1), y_pred, output_dict = True)
    # save classification report as csv
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join("out", "nn_report.csv"))

# a sequential neural network
def seq_nn_model(X_train_scaled, y_train, X_test_scaled, y_test):
    # initialize model
    model = Sequential()
    # set parameters 
    model.add(Dense(256, input_shape = X_train_scaled.shape[1]), activation="relu")
    model.add(Dense(128, activation="relu"))
    # classifier layer
    model.add(Dense(10, activation="softmax"))
    # define the gradient descent (learning rate)
    sgd = SGD(0.01)
    # compile model with additional parameters 
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd, 
                  metrics = ["accuracy"])
    # fit the training data and training labels
    model.fit(X_train_scaled, y_train,
                        validation_data = (X_test_scaled, y_test),
                        epochs = 10,
                        batch_size = 128)
    return model

# get predictions
def seq_nn_predictions(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled, batch_size = 32)
    # print classification report without additional parameter
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1)))
    # get classification report
    report = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), output_dict = True)
    # save classification report as csv
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join("out", "seq_nn_report.csv"))
    
def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-d", "--data", required=True, help="the dataset to use")
    args = vars(ap.parse_args())
    return args

# let's run the code
def main():
    args = parse_args()
    dataset = choose(args["data"])
    # if the user chooses mnist_784
    if dataset == "mnist_784":
        # do this
        X_train_scaled, X_test_scaled, y_train, y_test = load_and_process_mnist(dataset)
        nn = nn_model(X_train_scaled, y_train)
        nn_predictor(nn, X_test_scaled, y_test)
        model = seq_nn_model(X_train_scaled, y_train, X_test_scaled, y_test)
        seq_nn_predictions(model, X_test_scaled, y_test)
    # if not    
    else:
        # do this
        X_train_scaled, X_test_scaled, y_train, y_test = load_and_process_cifar10()
        nn = nn_model(X_train_scaled, y_train)
        nn_predictor(nn, X_test_scaled, y_test)
        model = seq_nn_model(X_train_scaled, y_train, X_test_scaled, y_test)
        seq_nn_predictions(model, X_test_scaled, y_test)
        
if __name__ == "__main__":
    main()


