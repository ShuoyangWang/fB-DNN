#!/usr/bin/env python


# coding: utf-8


"""

 

We run Lassonet over a custom dataset.

 

This dataset should be formatted similar to the Mice Protein Expression dataset.

 

Each feature should represent the expression level of one protein.

 

"""

import torch
import torch.nn.functional as F
import ast
import pdb
import pickle
import sys
import time
from calendar import EPOCH

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lassonet.interfaces import LassoNetClassifier, LassoNetClassifierCV
from lassonet.plot import plot_path
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale


def indices_to_groups(indices, J):
    result = []
    count = 0
    for i in indices:
        count += 1
        if count == J:
            result.append(i // J + 1)
            count = 0
    return tuple(result)


def arg_to_J(J):
    J = J[1:-1].split(",")
    return [int(x) for x in J]


def arg_to_dropout(dropout):
    dropout = dropout[1:-1].split(",")
    return [float(x) for x in dropout]


def arg_to_hidden_layers(hidden_layers):
    results = ast.literal_eval(hidden_layers)
    ret_val = []
    for result in results:
        ret_val.append(str(result)[1:-1])
    return ret_val


def get_features(J, max_J, p):
    columns = []
    for i in range(p):
        start = max_J * i
        for j in range(J):
            columns.append(start + j)
    return columns


def dump_to_csv(X, J, max_J, p):
    filename = "X_train_p{}_J{}_max_J{}.csv".format(p, J, max_J)
    df = pd.DataFrame(X)
    df.to_csv(filename)


def get_train_test(work_dir, task_id, J, max_J, p):
    stringX1 = work_dir + "/X"
    stringy1 = work_dir + "/y"
    file_type = ".csv"

    stringX = stringX1 + task_id + file_type
    stringy = stringy1 + task_id + file_type

    X = pd.read_csv(stringX, header=None, sep=",")
    y = pd.read_csv(stringy, header=None, dtype=int)

    X = X.values
    y = y.values
    y = y.flatten()

    # Adjust X according to J and max_J
    X = X[:, get_features(J, max_J, p)]
    # Shape should be (?, p * J)

    # Split your data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # dump_to_csv(X_train, J, max_J, p)
    return X_train, X_test, y_train, y_test


def get_validation_holdout(work_dir, task_id, J, max_J, p):
    X_train, X_test, y_train, y_test = get_train_test(work_dir, task_id, J, max_J, p)

    # Splitting test data into 20% validation and 20% holdout data
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    return X_val, X_holdout, y_val, y_holdout




def classify(work_dir, task_id, J, max_J, p, hidden_layers, path, M, dropout):
    start_time = time.time()

    X_train, X_test, y_train, y_test = get_train_test(work_dir, task_id, J, max_J, p)
    X_val, X_holdout, y_val, y_holdout = get_validation_holdout(
        work_dir, task_id, J, max_J, p
    )

    J = int(J)
    p = int(p)

    # Feature groups: arbitrarily assigned
    groups = [list(range(i, i + J)) for i in range(0, p * J, J)]

    # n_elements = p * J

    # X_train = X_train[:, :n_elements]
    # X_test = X_test[:, :n_elements]
    # X_val = X_val[:, :n_elements]
    # X_holdout = X_holdout[:, :n_elements]

    hidpar = str(hidden_layers)
    # Define the classifier
    model = LassoNetClassifierCV(
        hidden_dims=tuple(
            map(int, hidpar.split(","))
        ),  # (100,), (100,100),(100,50,100)
        groups=groups,
        lambda_start="auto",
        gamma=0.0,
        path_multiplier=float(
            path
        ),  # determines how rapidly we increase the penalty at each step
        M=float(M),  # hierarchy parameter
        dropout=dropout,
        batch_size=20,
        n_iters=(1000, 100),
        patience=(100, 10),
        tol=0.99,
        verbose=1,
    )

 

   
    # Fit the model
    path = model.path(X_train, y_train)


    # Calculate the testing error
    y_val_pred = model.predict(X_val)
    testing_error = 1 - accuracy_score(y_val, y_val_pred)

    n_selected = []
    n_selected_elements = []
    accuracy = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        y_pred = model.predict(X_train)
        n_selected.append(save.selected.sum())
        n_selected_elements.append(save.selected)
        accuracy.append(accuracy_score(y_train, y_pred))
        lambda_.append(save.lambda_)
    
    y_test_tensor = torch.tensor(y_val, dtype=torch.long)

   
    probabilities = model.predict_proba(X_val)
    predictions = model.predict(X_val)
    
    # Convert probabilities to tensor
    probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)

    # Use PyTorch's cross entropy loss function (it combines LogSoftmax and NLLLoss)
    loss_fn = torch.nn.CrossEntropyLoss()
    cross_entropy_loss = loss_fn(probabilities_tensor, y_test_tensor)




    end_time = time.time()
    computational_time = end_time - start_time
    n_selected_indices = tuple(n_selected_elements[-1].nonzero().squeeze().tolist())
    selected_groups = indices_to_groups(n_selected_indices, J)
    ret_val = {
        "Selected feature groups": selected_groups,
        "Computational time": computational_time,
        "Testing Error": testing_error,
        "Cross Entropy Loss": cross_entropy_loss.item()
    }

    return ret_val


if __name__ == "__main__":
    index = str(sys.argv[1])
    J = int(sys.argv[2])
    p = int(sys.argv[3])
    hidden_layers = str(sys.argv[4])
    path = float(sys.argv[5])
    M = int(sys.argv[6])
    dropout = float(sys.argv[7])
    max_J = int(sys.argv[8])
    workdir = str(sys.argv[9])

    results = []
    

    result = classify(workdir, index, J, max_J, p, hidden_layers, path, M, dropout)
    results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results" + "_" + index + ".csv", index=False)
