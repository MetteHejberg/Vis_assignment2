## 1. Assignment 2 - Image classifier benchmark scripts
For this assignment, you will take the classifier pipelines we covered in lecture 7 and turn them into *two separate ```.py``` scripts*. Your code should do the following:

- One script should be called ```logistic_regression.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Logistic Regression model using ```scikit-learn```
  - Print the classification report to the terminal **and** save the classification report to ```out/lr_report.txt```
- Another scripts should be called ```nn_classifier.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Neural Network model using the premade module in ```neuralnetwork.py```
  - Print output to the terminal during training showing epochs and loss
  - Print the classification report to the terminal **and** save the classification report to ```out/nn_report.txt```

## 2. Methods
This repository contains two scripts that classify images, one using a logistic regression classifier and one using two different neural networks

```logistic_regression_classifier.py``` performs logistic regression to classify images

```nn_classifier.py``` uses two different neural networks to classify images. The first neural network is simpler and the second is a sequential neural network with a hidden layer. 

## 3.1 Usage ```logistic_regression.py``` 
To run the code:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/logistic_regression.py -d "the data to use"```
  - The data to use can either be "mnist_784" of "cifar10"

## 3.2 Usage ```nn_classifier.py```
To run the code:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Place the ```utils``` folder inside the ```src``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/nn_classifier.py -d "the data to use"```
  - The data to use can either be "mnist_784" of "cifar10"

## 4. Discussion of Results

