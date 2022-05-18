# path tools
import os
import sys

# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

import pandas as pd

import argparse

# a function that allows that user to choose which dataset they want to work with - either "mnist_784" ot "cifar10"
def choose(data):
    dataset = data
    return dataset

# if the user chooses cifar10, this function will run 
def load_and_process_cifar10():
    # load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # define the labels - they will not appear in the classification report
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

# if the user chooses mnist_784, this function will run
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
                                                    train_size=7500) 
   
    # normalize
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255
    return X_train_scaled, X_test_scaled, y_train, y_test

# define the classifier
def classifier(X_train_scaled, y_train, X_test_scaled, y_test):
    # set parameters                                                    
    clf = LogisticRegression(penalty = "none",
                         tol = 0.1,
                         solver = "saga",
                         multi_class = "multinomial").fit(X_train_scaled, y_train)
    # get predictions                                                    
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    print(report)
    # create outpath 
    p = os.path.join("out", "lr_report.txt")
    # save classification report
    sys.stdout = open(p, "w")
    text_file = print(report)
    
def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-d", "--data", required=True, help="the dataset to use")
    args = vars(ap.parse_args())
    return args    
    
# let's get the code to run!
def main():
    args = parse_args()
    dataset = choose(args["data"])
    # if the user chooses mnist_784                                                    
    if dataset == "mnist_784":
        # do this                                                
        X_train_scaled, X_test_scaled, y_train, y_test = load_and_process_mnist(dataset)
        classifier(X_train_scaled, y_train, X_test_scaled, y_test)
    # if not                                                    
    else:
        # do this                                                 
        X_train_scaled, X_test_scaled, y_train, y_test = load_and_process_cifar10()
        classifier(X_train_scaled, y_train, X_test_scaled, y_test)
    
if __name__ == "__main__":
    main()




