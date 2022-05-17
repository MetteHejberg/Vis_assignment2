## 1. Assignment 2 - Image classifier benchmark scripts
Link to repository: https://github.com/MetteHejberg/Vis_assignment2
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

```logistic_regression_classifier.py``` performs logistic regression to classify images from scikit learn. I use Open-cv to convert the images to arrays, convert to grayscale, normalzie. The script furthermore reshapes the images fra cifar10 to 1-d arrays, whereas the mnist images are already 1-d. Then the script intializes the logical regression classifier, gets the classification report and saves the report to ```out``` 

```nn_classifier.py``` uses two different neural networks to classify images. The preprocessing of the images are the same as in the previous script. The first neural network is simpler and prewritten (from the ```utils``` folder) and the second is a sequential neural network with a hidden layer. Both models creates a classification report which is saved in ```out``` 

Since both scripts work with both the cifar10 dataset and mnist and the cifar10 dataset has labels whereas mnist does not (because the pictures are numbers, which correspond to the default labels in the classification report), I provide the list of labels and corresponding number in the classification report for the cifar10 dataset:
- airplane = 0
- automobile = 1
- bird = 2
- cat = 3
- deer = 4
- dog = 5
- frog = 6
- horse = 7
- ship = 8
- truck = 9

## 3.1 Usage ```logistic_regression.py``` 
To run the code:
- Pull this repository with this file structure
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/logistic_regression.py -d "the data to use"```
  - The data to use can either be "mnist_784" of "cifar10"
  - ```lr_report.txt``` in ```out``` was created with: ```python src/logistic_regression.py -d "mnist_784"```

## 3.2 Usage ```nn_classifier.py```
To run the code:
- Pull this repository with this file structure
- Place the ```utils``` folder inside the ```src``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/nn_classifier.py -d "the data to use"```
  - The data to use can either be "mnist_784" of "cifar10"

## 4. Discussion of Results
Since the mnist images are 1-d array, I reduced the cifar10 images to 1-d aswell, so I could use the same model for both datasets and in that way compare the model's performance. Both the logistic regression classifier and both neural networks perform significantly better on the mnist dataset, and in fact, perhaps a bit surprisingly, the logistic regression classifier performs better than the neural networks (__still true?__). You would probably getter a better performance with something like vgg16 on cifar10 which can take 2-d arrays and 3 color channels (see assignmen 3), which wouldn't work for the mnist dataset.

It seems that it is difficult to write a single, successful model for two very different datasets and the results definitely reflect this.
