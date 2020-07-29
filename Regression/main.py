import csv
from datetime import datetime
from math import sqrt
import statistics

import keras
import keras.layers as layers
import numpy as np
from keras.layers import Dropout
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd

from dicts import SetOfFeatures
from preprocessing import import_data
from models import do_grid_search, create_model, train_and_evaluate_models, write_price_differences, manual_gridsearch_svr, support_vector_machine, linear_regression, random_forrest

# region Predefining

FEATURE_SET = SetOfFeatures.all_features
TRAIN_SPLIT = 0.7


FILE_PATH = '/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# fix random seed for reproducibility

np.random.seed(5)

# endregion


#endregion

# region different models

# region linear regression tests

def test_linear_regression(loga=False, norm=False):
    train_split = TRAIN_SPLIT
    feature_set = FEATURE_SET
    logarithm = loga
    normalize = norm
    name = 'name'

    parameter_dict = {
        'name': name,
        'feature_set': feature_set,
        'logarithm': logarithm,
        'normalize': normalize,
        'train_split': train_split,
    }

    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        logarithm, normalize, FILE_PATH, TRAIN_SPLIT, FEATURE_SET,)

    prediction = linear_regression(x_tr, y_tr, x_te, L2_matrix, parameter_dict)
    write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

# endregion

# region SVR tests
def test_svr():
    train_split = TRAIN_SPLIT
    feature_set = FEATURE_SET
    logarithm = True
    normalize = True
    name = 'name'

    parameter_dict = {
        'name': name,
        'feature_set': feature_set,
        'logarithm': logarithm,
        'normalize': normalize,
        'train_split': train_split,
    }

    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        logarithm, normalize, FILE_PATH, TRAIN_SPLIT, FEATURE_SET)

    prediction = support_vector_machine(x_tr, y_tr, x_te, L2_matrix, 1e-06, 300, 0.01, parameter_dict)
    write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

def test_svr_grid():
    train_split = TRAIN_SPLIT
    feature_set = FEATURE_SET
    logarithm = False
    normalize = False
    name = 'name'

    parameter_dict = {
        'name': name,
        'feature_set': feature_set,
        'logarithm': logarithm,
        'normalize': normalize,
        'train_split': train_split,
    }

    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        logarithm, normalize, FILE_PATH, TRAIN_SPLIT, FEATURE_SET)

    epsilon_list = [1e-5]
    gamma_list = [200]
    c_list = [0.01]

    manual_gridsearch_svr(x_tr, y_tr, x_te, y_te, L2_matrix, epsilon_list, gamma_list, c_list, parameter_dict)
# endregion

# region random forest tests
def test_random_forest():
    train_split = TRAIN_SPLIT
    feature_set = FEATURE_SET
    logarithm = True
    normalize = False
    name = 'logarithm'

    parameter_dict = {
        'name': name,
        'feature_set': feature_set,
        'logarithm': logarithm,
        'normalize': normalize,
        'train_split': train_split,
    }

    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        logarithm, normalize, FILE_PATH, TRAIN_SPLIT, FEATURE_SET)
    for i in range(3):
        prediction = random_forrest(x_tr, y_tr, x_te, L2_matrix, parameter_dict)
        write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

# endregion

# region deep learning tests
def test_deep_learning():
    epochs = 5000
    lr = 0.001
    batch_size = 16
    validation_split = 0.2
    optimizer = "adam"
    logarithm = False
    normalize = False
    batchnorm = False
    dropout = False
    constant = True
    name = 'normal'
    train_split = TRAIN_SPLIT
    parameter_dict = {
        'name': name,
        'feature_set': FEATURE_SET,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'optimizer': optimizer,
        'logarithm': logarithm,
        'normalize': normalize,
        'batchnorm': batchnorm,
        'train_split': train_split,
        'dropout': dropout,
        'constant': constant
    }
    # endregion

    name = 'feature_sets'
    lr = 0.01
    epochs = 1900
    dropout = True
    constant = True
    logarithm = False

    parameter_dict = {
            'name': name,
            'feature_set': FEATURE_SET,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'optimizer': optimizer,
            'logarithm': logarithm,
            'normalize': normalize,
            'batchnorm': batchnorm,
            'train_split': train_split,
            'dropout': dropout,
            'constant': constant
        }
    train_n_test_model(2, parameter_dict)



def train_n_test_model(times, parameters):
    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        parameters['logarithm'], parameters['normalize'], FILE_PATH, parameters['train_split'], parameters['feature_set'])

    for i in range(times):
        prediction = train_and_evaluate_models(x_tr, y_tr, x_te, L2_matrix, parameters)
        write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameters)
# endregion

# region deep learning gridsearch tests

def test_deep_learning_GS(epochs, lr, constant, dropout):
    # region data import
    train_split = TRAIN_SPLIT
    feature_set = FEATURE_SET
    logarithm = True
    normalize = False
    name = 'with_logarithm'

    parameter_dict = {
        'name': name,
        'feature_set': feature_set,
        'logarithm': logarithm,
        'normalize': normalize,
        'train_split': train_split,
    }

    x_tr, y_tr, x_te, y_te, L2_matrix, saved_data, scaler = import_data(
        logarithm, normalize, FILE_PATH, TRAIN_SPLIT, FEATURE_SET)

    # endregion

    # region parameters
    learning_rate_list = [lr]
    epochs_list = [epochs]
    batch_size_list = [16]
    hidden_layers_list = [4]
    number_neurons_list = [256]
    batchnorm_list = [False]
    dropout_list = [dropout]
    init_mode_list = ['he_normal']
    input_dim = [x_tr.shape[1]]
    optimizer_list = ['adam']
    dropout_rate = [0.1]
    constant = [constant]
    set_of_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    parameters = dict(batch_size=batch_size_list, epochs=epochs_list, hidden_layers=hidden_layers_list,
                      neurons=number_neurons_list, learn_rate=learning_rate_list, batchnormalize=batchnorm_list,
                      init_mode=init_mode_list, input_dimension=input_dim, optimizer=optimizer_list,
                      dropout=dropout_list, dropout_rate=dropout_rate, constant=constant)

    # endregion

    do_grid_search(x_tr, y_tr, parameters, create_model, parameter_dict)

# endregion


# endregion


# region tests
# test_linear_regression(False, False)


#test_svr()
#test_svr_grid()

#test_random_forest()

test_deep_learning()
#test_deep_learning_GS(epochs=1900, lr=0.01, constant=True, dropout=True)
#test_deep_learning_GS(epochs=2500, lr=0.1, constant=False, dropout=False)

#test_deep_learning_GS(epochs=2400, lr=0.001, number_neurons=256)

# endregion
