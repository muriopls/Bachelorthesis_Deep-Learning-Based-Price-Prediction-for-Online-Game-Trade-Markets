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


from dicts import SetOfFeatures
from preprocessing import import_data

# region Predefining
experiment_time = datetime.now().strftime("%Y%m%d_%H%M")
export_to_disk = False
big_export_path_file = ""
export_path_file = ""
if export_to_disk:
    big_export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "rmse")
else:
    big_export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/price_differences/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/optimization/{}_{}.csv".format(
        experiment_time, "rmse")

selected_features = SetOfFeatures.all_features
epochs = 500
lr = 0.01
batch_size = 16
validation_split = 0.2
optimizer = "adam"
logarithm = False
normalize = False
batchnorm = False
exponent = False
dropout = False
train_split = 0.7
file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
name = 'epochs_determination'
parameter_dict = {
    'name': name,
    'feature_set': selected_features,
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'validation_split': validation_split,
    'optimizer': optimizer,
    'logarithm': logarithm,
    'normalize': normalize,
    'batchnorm': batchnorm,
    'exponent': exponent,
    'train_split': train_split,
    'file_path': file_path,
    'dropout': dropout
}
x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(
    logarithm, normalize, file_path, train_split, exponent, selected_features)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# endregion


# region different models

def add_layer(model, neurons, activation="relu", init_mode="glorot_uniform", batchnormalize=False):

    model.add(layers.Dense(neurons, use_bias=False, kernel_initializer=init_mode))

    if batchnormalize:
        model.add(layers.BatchNormalization())

    model.add(layers.Activation(activation))

    if dropout:
        model.add(Dropout(0.2))

def create_model(hidden_layers=7, neurons=64, learn_rate=0.01, opt="Adam", activation="relu", init_mode="glorot_uniform", batchnormalize=False):
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False, kernel_initializer=init_mode))

    if batchnormalize:
        model.add(layers.BatchNormalization())

    for i in range(1, hidden_layers+1):
        if (neurons/(2**(i-1))) < 2:
            add_layer(model, 2, activation, init_mode, batchnormalize)
        else:
            add_layer(model, int(neurons/(2**(i-1))), activation, init_mode, batchnormalize)

    model.add(layers.Dense(1, use_bias=False, kernel_initializer=init_mode))

    model.compile(optimizer=opt, lr=learn_rate, loss='mean_absolute_error')

    return model

def create_model_down(hidden_layers=7, neurons=64):
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    for i in range(hidden_layers):
        if (neurons/(2**(i-1))) < 2:
            add_layer(model, 2)
        else:
            add_layer(model, int(neurons/(2**(i-1))))

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model_up(hidden_layers=7, neurons=64):
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    for i in range(hidden_layers):
        add_layer(model, int(neurons/(neurons/(2**i))))

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

# region layer numbers
def create_model1():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model2():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model3():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model4():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)


    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model5():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model6():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model7():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model8():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model9():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model10():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)
    add_layer(model, 64)

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

# endregion

#region different architectures
def create_model_down():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(128, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(32, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(16, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(8, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(4, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(2, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model_constant():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model

def create_model_up():
    model = keras.models.Sequential()

    # region create and train model
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(2, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(4, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(8, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(16, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(32, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(128, use_bias=False))

    if dropout:
        model.add(Dropout(0.2))

    model.add(layers.Dense(1, use_bias=False))

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    return model
#endregion

def train_and_evaluate_models(batchnorm, x_train, y_train, x_test, L2, optimizer, test_name, to_disk):
    model = keras.models.Sequential()

    # region create and train model
    print('create neural network')
    model.add(layers.Dense(x_tr.shape[1], use_bias=False))

    add_layer(model, 256)
    add_layer(model, 128)
    add_layer(model, 64)
    add_layer(model, 32)
    add_layer(model, 16)

    model.add(layers.Dense(1, use_bias=False))

    log_dir = ""

    to_disk = True
    if to_disk:
        log_dir = "E:\Bachelorarbeit_Informatik\Auswertung\logs\{}\{}_{}".format(
            experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name)
    else:
        log_dir = 'C:\\Users\\murio\\PycharmProjects\\Data\\pricePrediction\\logs\\{}\\{}_{}'.format(
            experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name)

    print(log_dir)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True, write_images=True)

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    print('train neural network')
    history = model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split,  callbacks=[tensorboard], verbose=2)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list

#region other approaches

def support_vector_machine(x_train, y_train, x_test, L2, e, g, C):
    model = SVR(C=C, epsilon=e, gamma=g)
    # model = SVR()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list


def manual_gridsearch_svr(x_train, y_train, x_test, y_test, L2, epsilons, gammas, Cs):
    # region export header
    csv_header = []
    csv_header.append('Average Error')
    csv_header.append('Standard Deviation')
    csv_header.append('Epsilon')
    csv_header.append('Gamma')
    csv_header.append('C')

    with open(export_path_file, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)
    # endregion

    if logarithm:
        y_test = 10 ** y_test
    if normalize:
        y_test = y_test*L2[0]

    for epsilon in epsilons:
        print("Using now epsilon = %f" % epsilon)
        for gamma in gammas:
            print("Using now gamma = {gamma}".format(gamma=str(gamma)))
            for C in Cs:
                print("Using now C = %f" % C)
                prediction = support_vector_machine(x_train, y_train, x_test, L2, epsilon, gamma, C)

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
                with open(export_path_file, 'a',
                          newline='') as csvFile:
                    writer = csv.writer(csvFile, delimiter=';')
                    writer.writerow(row_container)
    return


def svc_param_selection(X, y):
    Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    epsilons = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    gammas = ['scale', 'auto']

    # Cs = [1e-3, 1e-2, 1e-1, 1]
    # epsilons = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # gammas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 'scale', 'auto']
    param_grid = {'C': Cs, 'epsilon': epsilons, 'gamma': gammas}
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, n_jobs=-1)
    grid_result = grid_search.fit(X, y)

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

    with open(export_path_file, 'a',
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
        with open(export_path_file, 'a',
                  newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerow(row_container)

    return


def linear_regression(x_train, y_train, x_test, L2):
    model = LinearRegression()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list


def random_forrest(x_train, y_train, x_test, L2):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list

#endregion

# endregion

def test_model(times):
    for i in range(times):
        prediction = train_and_evaluate_models(
            batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
        write_price_differences(prediction, x_te, y_te,
                                L2_matrix, saved_data, parameter_dict)

def write_price_differences(prediction_list, x_test, y_test_values, L2, saved_dataframe, parameter_dict):

    #region init exports
    y_test_list = y_test_values.values.tolist()

    if logarithm:
        y_test_values = 10 ** y_test_values
    if normalize:
        y_test_values = y_test_values*L2[0]

    deviation_list = []
    for i in range(len(y_test_values)):
        deviation_list.extend(abs(abs(prediction_list[i]) - abs(y_test_values.iloc[i])))

    print(max(deviation_list), sum(deviation_list) / len(deviation_list))
    print(len(deviation_list))

    print('write export file')
    index_list = x_test.index

    # inverse normalization
    if normalize:
        x_test.multiply(L2, axis=1)

    #endregion

    # region big export
    csv_header = []
    csv_header.append('Difference Absolute')
    csv_header.append('Actual')
    csv_header.append('Prediction')
    csv_header.extend(list(x_test))

    with open(big_export_path_file, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)

    for i in range(len(y_test_values)):
        # write csv file
        with open(big_export_path_file, 'a',
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

    with open(big_export_path_file, 'a',
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

    with open(export_path_file, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(csv_header)
    #endregion

    sum_deviation_list = sum(deviation_list)
    mean_absolute_error = int(sum_deviation_list / len(deviation_list))

    rms = sqrt(metrics.mean_squared_error(y_test_values, prediction_list))

    row_container = []
    row_container.append(int(mean_absolute_error))
    row_container.append(int(rms))

    for value in parameter_dict.values():
        row_container.append(value)

    # region testing export header rows
    with open(export_path_file, 'a',
              newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(row_container)

    # endregion
    # endregion

def do_grid_search(x_train, y_train, param_grid, model_func):
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

    with open(export_path_file, 'a',
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
        with open(export_path_file, 'a',
                  newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerow(row_container)

    return

# region parameters
learning_rate_list = [0.001, 0.0001, 0.00001]

epochs_list = [1000]
batch_size_list = [16]
hidden_layers_list = [5]
number_neurons_list = [256]
batchnorm_list = [True, False]

init_mode_list = ['lecun_uniform', 'glorot_normal', 'glorot_uniform']
optimizer_list = ["Adam", "RMSprop", "SGD"]

weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

set_of_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#endregion


# endregion
# for i in range(10):
#     prediction = train_and_evaluate_models(
#         batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)

print("Gridesearch Durchlauf 1")
experiment_time = datetime.now().strftime("%Y%m%d_%H%M")
export_to_disk = False
big_export_path_file = ""
export_path_file = ""
if export_to_disk:
    big_export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "rmse")
else:
    big_export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/price_differences/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/optimization/{}_{}.csv".format(
        experiment_time, "rmse")

parameters = dict(batch_size=batch_size_list, epochs=epochs_list, hidden_layers=hidden_layers_list, neurons = number_neurons_list, learn_rate=learning_rate_list, init_mode=init_mode_list, batchnormalize=batchnorm_list)
do_grid_search(x_tr, y_tr, parameters, create_model)


# do_grid_search(x_tr, y_tr, parameters, create_model9)
# do_grid_search(x_tr, y_tr, parameters, create_model10)
# do_grid_search(x_tr, y_tr, parameters, create_model_up)

# svc_param_selection(x_tr, y_tr)

# # region SVR parameter tuning
# Cs = [1e-2, 1e-1, 1, 10, 100, 1000]
# epsilons = [1e-5, 1e-3, 1e-1, 1, 100]
# gammas = [1e-5, 1e-3, 1e-1, 1, 100, 'scale', 'auto']
#
# Cs = [16, 32, 48]
# epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# gammas = [1]
#
#
# manual_gridsearch_svr(x_tr, y_tr, x_te, y_te, L2_matrix, epsilons, gammas, Cs)
#
# # endregion
#
# prediction = support_vector_machine2(x_tr, y_tr, x_te, L2_matrix)
# write_price_differences(prediction, x_te, y_te,
#                         L2_matrix, saved_data, parameter_dict)

#print(svc_param_selection(x_tr, y_tr))

""" # normal
for i in range(3):
    prediction = train_and_evaluate_models(
        batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
    write_price_differences(prediction, x_te, y_te,
                            L2_matrix, saved_data, parameter_dict)


# modified data-set
file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified_removed_outliers.csv'
x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(
    logarithm, normalize, file_path, train_split, exponent, selected_features)
name = "without_top_15"
parameter_dict = {
    'name': name,
    'feature_set': selected_features,
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'validation_split': validation_split,
    'optimizer': optimizer,
    'logarithm': logarithm,
    'normalize': normalize,
    'batchnorm': batchnorm,
    'exponent': exponent,
    'train_split': train_split,
    'file_path': file_path,
    'dropout': dropout
}
for i in range(3):
    prediction = train_and_evaluate_models(
        batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
    write_price_differences(prediction, x_te, y_te,
                            L2_matrix, saved_data, parameter_dict)
 """

# for item in learning_rate_list:
#     lr = item
#     name = "lr_" + str(item)
#     parameter_dict = {
#         'name': name,
#         'feature_set': selected_features,
#         'epochs': epochs,
#         'lr': lr,
#         'batch_size': batch_size,
#         'validation_split': validation_split,
#         'optimizer': optimizer,
#         'logarithm': logarithm,
#         'normalize': normalize,
#         'batchnorm': batchnorm,
#         'exponent': exponent,
#         'train_split': train_split,
#         'file_path': file_path,
#         'dropout': dropout
#     }
#     test_model(6)


# normal 1500 epochs

# for i in range(3):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)


# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
#
# optimizer = "RMSprop"
#
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(
#     logarithm, normalize, file_path, train_split, exponent, selected_features)
#
# name = "RMSprop"
# parameter_dict = {
#     'name': name,
#     'feature_set': selected_features,
#     'epochs': epochs,
#     'lr': lr,
#     'batch_size': batch_size,
#     'validation_split': validation_split,
#     'optimizer': optimizer,
#     'logarithm': logarithm,
#     'normalize': normalize,
#     'batchnorm': batchnorm,
#     'exponent': exponent,
#     'train_split': train_split,
#     'file_path': file_path,
#     'dropout': dropout
# }

# for i in range(6):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)

# test_model(3)

# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
#
# optimizer = 'SGD'
#
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(
#     logarithm, normalize, file_path, train_split, exponent, selected_features)
#
# name = "SGD"
# parameter_dict = {
#     'name': name,
#     'feature_set': selected_features,
#     'epochs': epochs,
#     'lr': lr,
#     'batch_size': batch_size,
#     'validation_split': validation_split,
#     'optimizer': optimizer,
#     'logarithm': logarithm,
#     'normalize': normalize,
#     'batchnorm': batchnorm,
#     'exponent': exponent,
#     'train_split': train_split,
#     'file_path': file_path,
#     'dropout': dropout
# }

#
# for i in range(6):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)


# test_model(3)

# for i in range(10):
#     prediction = random_forrest(x_tr, y_tr, x_te, L2_matrix)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)

# for i in range(10):
#     prediction = linear_regression(x_tr, y_tr, x_te, L2_matrix)
#     write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)
#
# print(svc_param_selection(x_tr, y_tr))

# prediction = support_vector_machine(x_tr, y_tr, x_te, L2_matrix)
# write_price_differences(prediction, x_te, y_te,
#                             L2_matrix, saved_data, parameter_dict)
#
# # modified data-set 1500 epochs
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified_removed_outliers.csv'
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(
#     logarithm, normalize, file_path, train_split, exponent, selected_features)
# epochs = 5000
# name = "without_top_15_5000_Epochs"
# parameter_dict = {
#     'name': name,
#     'feature_set': selected_features,
#     'epochs': epochs,
#     'lr': lr,
#     'batch_size': batch_size,
#     'validation_split': validation_split,
#     'optimizer': optimizer,
#     'logarithm': logarithm,
#     'normalize': normalize,
#     'batchnorm': batchnorm,
#     'exponent': exponent,
#     'train_split': train_split,
#     'file_path': file_path,
#     'dropout': dropout
# }


# for item in epochs_list:
#     lr = item
#     name = "epochs: " + item
#     for i in range(10):
#         prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#         write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#

# for item in batch_size_list:
#     batch_size = item
#     name = "batch-size_" + str(item)
#     parameter_dict = {
#         'name': name,
#         'feature_set': selected_features,
#         'epochs': epochs,
#         'lr': lr,
#         'batch_size': batch_size,
#         'validation_split': validation_split,
#         'optimizer': optimizer,
#         'logarithm': logarithm,
#         'normalize': normalize,
#         'batchnorm': batchnorm,
#         'exponent': exponent,
#         'train_split': train_split,
#         'file_path': file_path
#     }
#     for i in range(10):
#         prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#         write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

# for i in range(10):
#     prediction = random_forrest(x_tr, y_tr, x_te, L2_matrix)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
