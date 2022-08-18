#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:50:41 2020

@author: zhe
"""

# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np
import math


# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    x = np.loadtxt(filename, usecols=(range(102)))
    y = np.loadtxt(filename, usecols=102)
    return x, y


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    # your code
    split_length = len(x) * train_proportion
    x_train, x_test = np.split(x, [int(split_length)])
    y_train, y_test = np.split(y, [int(split_length)])
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation, check our lecture slides
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    # your code
    x = np.array(x)
    y = np.array(y)
    beta = np.linalg.inv(x.T.dot(x) + lambdaV*np.identity(x.shape[1])).dot(x.T.dot(y))
    return beta



# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    loss = 0
    loss = (1/len(y))*np.sum(np.square(y-y_predict))
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    # your code
    y_predict = np.dot(x,theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
    valid_losses = []
    training_losses = []
    valid_losses_one = []
    training_losses_one = []
    valid_losses_two = []
    training_losses_two = []
    valid_losses_three = []
    training_losses_three = []
    valid_losses_four = []
    training_losses_four = []
    first_fold = math.floor(int((1/4)*len(x_train))) 
    second_fold = math.floor(int((2/4)*len(x_train)))
    third_fold = math.floor(int((3/4)*len(x_train)))
        
    training_set_x_one = np.concatenate((x_train[first_fold:second_fold], x_train[second_fold:third_fold], x_train[third_fold:]))
    training_set_y_one = np.concatenate((y_train[first_fold:second_fold], y_train[second_fold:third_fold], y_train[third_fold:]))

    training_set_x_two = np.concatenate((x_train[:first_fold], x_train[second_fold:third_fold], x_train[third_fold:]))
    training_set_y_two = np.concatenate((y_train[:first_fold], y_train[second_fold:third_fold], y_train[third_fold:]))

    training_set_x_three = np.concatenate((x_train[:first_fold], x_train[first_fold:second_fold], x_train[third_fold:]))
    training_set_y_three = np.concatenate((y_train[:first_fold], y_train[first_fold:second_fold], y_train[third_fold:]))

    training_set_x_four = np.concatenate((x_train[:first_fold], x_train[first_fold:second_fold], x_train[second_fold:]))
    training_set_y_four = np.concatenate((y_train[:first_fold], y_train[first_fold:second_fold], y_train[second_fold:]))

    for i in range(len(lambdas)):
        beta = normal_equation(training_set_x_one, training_set_y_one, lambdas[i])
        predict_y = predict(training_set_x_one, beta)
        valid_y = predict(x_train[:first_fold], beta)
        train_loss = get_loss(training_set_y_one, predict_y)
        valid_loss = get_loss(y_train[:first_fold], valid_y)
        training_losses_one.append(train_loss)
        valid_losses_one.append(valid_loss)

    for i in range(len(lambdas)):
        beta = normal_equation(training_set_x_two, training_set_y_two, lambdas[i])
        predict_y = predict(training_set_x_two, beta)
        valid_y = predict(x_train[first_fold:second_fold], beta)
        train_loss = get_loss(training_set_y_two, predict_y)
        valid_loss = get_loss(y_train[first_fold:second_fold], valid_y)
        training_losses_two.append(train_loss)
        valid_losses_two.append(valid_loss)

    for i in range(len(lambdas)):
        beta = normal_equation(training_set_x_three, training_set_y_three, lambdas[i])
        predict_y = predict(training_set_x_three, beta)
        valid_y = predict(x_train[second_fold:third_fold], beta)
        train_loss = get_loss(training_set_y_three, predict_y)
        valid_loss = get_loss(y_train[second_fold:third_fold], valid_y)
        training_losses_three.append(train_loss)
        valid_losses_three.append(valid_loss)

    for i in range(len(lambdas)):
        beta = normal_equation(training_set_x_four, training_set_y_four, lambdas[i])
        predict_y = predict(training_set_x_four, beta)
        valid_y = predict(x_train[third_fold:], beta)
        train_loss = get_loss(training_set_y_four, predict_y)
        valid_loss = get_loss(y_train[third_fold:], valid_y)
        training_losses_four.append(train_loss)
        valid_losses_four.append(valid_loss)
    #For loop
    # Training for each lamda
    # go through normal equation for beta
    # predict for training and valid
    # need the loss

    for i in range(len(lambdas)):
        training_losses.append((training_losses_one[i] + training_losses_two[i] + training_losses_three[i] + training_losses_four[i])/4)

    for i in range(len(lambdas)):
        valid_losses.append((valid_losses_one[i] + valid_losses_two[i] + valid_losses_three[i] + valid_losses_four[i])/4)

    # your code
    return np.array(valid_losses), np.array(training_losses)


    
# Calcuate the l2 norm of a vector    
def l2norm(vec):
    # your code 
    norm = np.sqrt(np.dot(vec,vec.T))
    return norm

#  show the learnt values of Î² vector from the best Î»

def bar_plot(beta):
    #your code
    bar_plot = []
    for i in range(len(beta)):
        bar_plot.append(i)
    plt.bar(bar_plot,beta)
    plt.show()

    
    

if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss") 
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]


    # step 2: analysis 
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = l2norm(normal_beta)# your code get l2 norm of normal_beta
    best_beta_norm = l2norm(best_beta)# your code get l2 norm of best_beta
    large_lambda_norm = l2norm(large_lambda_beta)# your code get l2 norm of large_lambda_beta
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    
    
    # step 3: visualization
    bar_plot(best_beta)


    
