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

# region export file initialization
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
BIG_EXPORT_PATH_FILE = "C:/Users/murio/PycharmProjects/Data/pricePrediction/price_differences/{}_{}.csv".format(
    EXPERIMENT_TIME, "price_differences")
EXPORT_PATH_FILE = "C:/Users/murio/PycharmProjects/Data/pricePrediction/optimization/{}_{}.csv".format(
    EXPERIMENT_TIME, "rmse")
# endregion

# region Deep Learning

def add_layer(model, neurons, activation="relu", init_mode="glorot_uniform", batchnormalize=False, dropout=False):

    model.add(layers.Dense(neurons, use_bias=False, kernel_initializer=init_mode))

    if batchnormalize:
        model.add(layers.BatchNormalization())

    model.add(layers.Activation(activation))

    if dropout:
        model.add(Dropout(0.2))

def create_model(input_dimension, batch_size, epochs, hidden_layers, neurons, learn_rate, batchnormalize, init_mode):
    model = keras.models.Sequential()

    model.add(layers.Dense(input_dimension, use_bias=False, activation='relu'))

    # Implementation of a decreasing number of neurons in the hidden layers by a factor of 2
    for i in range(1, hidden_layers+1):
        if (neurons/2**(i-1)) > 1:
            add_layer(model, int(neurons / (2 ** (i-1))), 'relu', init_mode, batchnormalize)
        else:
            add_layer(model, 2, batchnormalize=batchnormalize)

        model.add(layers.Dense(1, use_bias=False))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_models(batchnorm, x_train, y_train, x_test, L2, parameter_dict):

    # region dict to variable transformation - just to make the code easier to read
    use_batchnormalization = parameter_dict['batchnorm']
    use_dropout = parameter_dict['dropout']
    optimizer = parameter_dict['optimizer']
    lr = parameter_dict['lr']
    validation_split = parameter_dict['validation_split']
    batch_size = parameter_dict['batch_size']
    epochs = parameter_dict['epochs']
    use_logarithm = parameter_dict['logarithm']
    use_normalization = parameter_dict['normalize']
    # endregion

    # region create model
    model = keras.models.Sequential()

    model.add(layers.Dense(x_train.shape[1], use_bias=False, activation='relu'))

    add_layer(model, 256, use_batchnormalization, dropout=use_dropout)
    add_layer(model, 128, use_batchnormalization, dropout=use_dropout)
    add_layer(model, 64, use_batchnormalization, dropout=use_dropout)
    add_layer(model, 32, use_batchnormalization, dropout=use_dropout)
    add_layer(model, 16, use_batchnormalization, dropout=use_dropout)

    model.add(layers.Dense(1, use_bias=False))

    # endregion

    # region Tensorboard initialization
    log_dir = "E:\Bachelorarbeit_Informatik\Auswertung\logs\{}\{}_{}".format(
        EXPERIMENT_TIME, datetime.now().strftime("%d%m%Y_%H%M%S"), parameter_dict['name'])

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True, write_images=True)
    # endregion

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    history = model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split,  callbacks=[tensorboard], verbose=2)

    # prediction on the test set
    prediction_list = model.predict(x_test)

    # region undo logarithm and normalization
    if use_logarithm:
        prediction_list = 10 ** prediction_list
    if use_normalization:
        prediction_list = prediction_list * L2[0]
    # endregion

    return prediction_list

def do_grid_search(x_train, y_train, param_grid, model_func, parameter_dict):
    model = KerasRegressor(build_fn=model_func, verbose=2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train.to_numpy(), y_train.to_numpy())

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # region export header
    csv_header = []
    csv_header.append('Average Error')
    csv_header.append('Standard Deviation')


    for key in param:
        csv_header.append(key)

    for key in parameter_dict:
        csv_header.append(key)

    with open(EXPORT_PATH_FILE, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)
    # endregion

    for mean, stdev, param in zip(means, stds, params):
        row_container = []
        row_container.append(abs(int(mean)))
        row_container.append(int(stdev))
        for value in param.values():
            row_container.append(value)

        for value in parameter_dict.values():
            row_container.append(value)

        # region testing export header rows
        with open(EXPORT_PATH_FILE, 'a',
                  newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerow(row_container)

    return

# endregion

# region other approaches

def support_vector_machine(x_train, y_train, x_test, L2, e, g, C, parameter_dict):
    model = SVR(C=C, epsilon=e, gamma=g)
    # model = SVR()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if parameter_dict['logarithm']:
        prediction_list = 10 ** prediction_list
    if parameter_dict['normalize']:
        prediction_list = prediction_list * L2[0]

    return prediction_list


def manual_gridsearch_svr(x_train, y_train, x_test, y_test, L2, epsilons, gammas, Cs, parameter_dict):
    # region export header
    csv_header = []
    csv_header.append('Average Error')
    csv_header.append('Standard Deviation')
    csv_header.append('Epsilon')
    csv_header.append('Gamma')
    csv_header.append('C')

    with open(EXPORT_PATH_FILE, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)
    # endregion

    for epsilon in epsilons:
        print("Using now epsilon = %f" % epsilon)
        for gamma in gammas:
            print("Using now gamma = {gamma}".format(gamma=str(gamma)))
            for C in Cs:
                print("Using now C = %f" % C)
                prediction = support_vector_machine(x_train, y_train, x_test, L2, epsilon, gamma, C, parameter_dict)

                deviation_list = []
                for i in range(len(y_test)):
                    deviation_list.extend(abs(prediction[i] - y_test.iloc[i]))

                mae = int(statistics.mean(deviation_list))
                stdev = int(statistics.stdev(deviation_list))

                row_container = []
                row_container.append((int(mae)))
                row_container.append(int(stdev))
                row_container.append(epsilon)
                row_container.append(gamma)
                row_container.append(C)

                # region testing export header rows
                with open(EXPORT_PATH_FILE, 'a',
                          newline='') as csvFile:
                    writer = csv.writer(csvFile, delimiter=';')
                    writer.writerow(row_container)
    return


def linear_regression(x_train, y_train, x_test, L2, parameter_dict):
    model = LinearRegression()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if parameter_dict['logarithm']:
        prediction_list = 10 ** prediction_list
    if parameter_dict['normalize']:
        prediction_list = prediction_list * L2[0]

    return prediction_list


def random_forrest(x_train, y_train, x_test, L2, parameter_dict):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if parameter_dict['logarithm']:
        prediction_list = 10 ** prediction_list
    if parameter_dict['normalize']:
        prediction_list = prediction_list * L2[0]

    return prediction_list

# endregion



def write_price_differences(prediction_list, x_test, y_test_values, L2, saved_dataframe, parameter_dict):

    #region init exports
    y_test_list = y_test_values.values.tolist()

    if parameter_dict['logarithm']:
        y_test_values = 10 ** y_test_values
    if parameter_dict['normalize']:
        y_test_values = y_test_values * L2[0]

    deviation_list = []
    for i in range(len(y_test_values)):
        deviation_list.extend(abs(prediction_list[i] - y_test_values.iloc[i]))

    print(max(deviation_list), sum(deviation_list) / len(deviation_list))
    print(len(deviation_list))

    print('write export file')
    index_list = x_test.index

    #endregion

    # region big export
    csv_header = []
    csv_header.append('Difference Absolute')
    csv_header.append('Actual')
    csv_header.append('Prediction')
    csv_header.extend(list(x_test))

    with open(BIG_EXPORT_PATH_FILE, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)

    for i in range(len(y_test_values)):
        # write csv file
        with open(BIG_EXPORT_PATH_FILE, 'a',
                  newline='', encoding="utf-8") as csvFile:
            writer = csv.writer(csvFile, delimiter=';')

            feature_list = []
            feature_list.append(
                 abs(int(y_test_values.iloc[i].values[0])-int(prediction_list[i])))
            feature_list.append(int(y_test_values.iloc[i].values[0]))
            feature_list.append(int(prediction_list[i]))

            # re-create feature list
            for feature in list(x_test):
                feature_list.append(
                    saved_dataframe.get_value(index_list[i], feature))

            # write prediction and feature values
            writer.writerow(feature_list)

    average_deviation = int(sum(deviation_list) / len(deviation_list))
    total_deviation = (sum(deviation_list))

    with open(BIG_EXPORT_PATH_FILE, 'a',
              newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow((average_deviation, len(
            deviation_list), parameter_dict['name']))
    # endregion

    # region testing export

    #region export header
    csv_header = []
    csv_header.append('Average Error')
    csv_header.append('Metrics mean squarred error')

    for key in parameter_dict:
        csv_header.append(key)

    with open(EXPORT_PATH_FILE, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)
    #endregion


    mean_absolute_error = int(statistics.mean(deviation_list))
    rms = sqrt(metrics.mean_squared_error(y_test_values, prediction_list))

    row_container = []
    row_container.append(int(mean_absolute_error))
    row_container.append(int(rms))

    for value in parameter_dict.values():
        row_container.append(value)

    # region testing export header rows
    with open(EXPORT_PATH_FILE, 'a',
              newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(row_container)

    # endregion
    # endregion


