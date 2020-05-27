import csv
from datetime import datetime
from math import sqrt

import keras
import keras.layers as layers
from keras.layers import Dropout
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from dicts import SetOfFeatures
from preprocessing import import_data

# region Predefining


selected_features = SetOfFeatures.all_features
price_difference_path = 'C:/Users/murio/PycharmProjects/Data/pricePrediction/price_differences'
epochs = 150
lr = 0.001
batch_size = 32
validation_split = 0.3
logarithm = False
normalize = False
batchnorm = False
experiment_time = datetime.now().strftime("%Y%m%d_%H%M")
export_to_disk = False
big_export_path_file = ""
export_path_file = ""

if export_to_disk:
    big_export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(experiment_time, "rmse")
else:
    big_export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/price_differences/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "C:/Users/murio/PycharmProjects/Data/pricePrediction/optimization/{}_{}.csv".format(experiment_time, "rmse")

# region testing export header
csv_header = []
csv_header.append('Average Error')
csv_header.append('Metrics mean squarred error')
header_parameters = {
    'name': 1,
    'feature_set': 1,
    'epochs': 1,
    'lr': 1,
    'batch_size': 1,
    'validation_split': 1,
    'optimizer': 1,
    'logarithm': 1,
    'normalize': 1,
    'batchnorm': 1,
    'exponent': 1,
    'train_split': 1,
    'file_path': 1
}

for key in header_parameters:
    csv_header.append(key)

with open(export_path_file, 'a',
          newline='', encoding="utf-8") as csvFile:
    writer = csv.writer(csvFile, delimiter=';')
    writer.writerow(csv_header)

# endregion
# endregion


# region different models



def train_and_evaluate_models(batchnorm, x_train, y_train, x_test, L2, optimizer, test_name, to_disk):
    model = keras.models.Sequential()

    # region create and train model
    print('create neural network')
    model.add(layers.Dense(512, use_bias=False))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(256, use_bias=False))

    model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(128, use_bias=False))

    model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))

    model.add(Dropout(0.2))

    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(32, use_bias=False))

    model.add(layers.Dense(1, use_bias=False))

    log_dir = ""

    to_disk = False
    if to_disk:
        log_dir = "E:\Bachelorarbeit_Informatik\Auswertung\logs\{}\{}_{}".format(experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name)

    else:
        log_dir = "C:\\Users\\murio\\PycharmProjects\\Data\\pricePrediction\\logs\\{}\\{}_{}".format(experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name)

    print(log_dir)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True, write_images=True)

    model.compile(optimizer=optimizer, lr=lr, loss='mean_absolute_error')

    print('train neural network')
    history = model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=epochs, batch_size=batch_size, validation_split=validation_split,  callbacks=[tensorboard], verbose=2)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list

def support_vector_machine(x_train, y_train, x_test, L2):
    model = SVC()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list

def svc_param_selection(X, y):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

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
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    prediction_list = model.predict(x_test)

    if logarithm:
        prediction_list = 10 ** prediction_list
    if normalize:
        prediction_list = prediction_list * L2[0]

    return prediction_list

# endregion

def write_price_differences(prediction_list, x_test, y_test_values, L2, saved_dataframe, parameter_dict):

    y_test_list = y_test_values.values.tolist()

    if logarithm:
        y_test_values = 10 ** y_test_values
    if normalize:
        y_test_values = y_test_values*L2[0]

    deviation_list = []
    for i in range(len(y_test_values)):
         deviation_list.extend(abs(prediction_list[i] - y_test_values.iloc[i]))

    print(max(deviation_list), sum(deviation_list) / len(deviation_list))
    print(len(deviation_list))

    print('write export file')
    index_list = x_test.index

    # inverse normalization
    if normalize:
        x_test.multiply(L2, axis=1)

    # region big export
    csv_header = []
    csv_header.append('Difference Absolute')
    csv_header.append('Actual')
    csv_header.append('Prediction')
    csv_header.append('Name')
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
            feature_list.append(abs(int(y_test_values.iloc[i].values[0])-int(prediction_list[i])))
            feature_list.append(int(y_test_values.iloc[i].values[0]))
            feature_list.append(int(prediction_list[i]))

            # re-create feature list
            for feature in list(x_test):
                feature_list.append(saved_dataframe.get_value(index_list[i], feature))

            # write prediction and feature values
            writer.writerow(feature_list)

    average_deviation = int(sum(deviation_list) / len(deviation_list))
    total_deviation = (sum(deviation_list))

    with open(big_export_path_file, 'a',
              newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow((average_deviation, len(deviation_list), parameter_dict['name']))
    #endregion

    #region testing export

    sum_deviation_list = sum(deviation_list)
    mean_squared_error = int(sum_deviation_list / len(deviation_list))

    rms = sqrt(metrics.mean_squared_error(y_test_values, prediction_list))

    row_container = []
    row_container.append(int(mean_squared_error))
    row_container.append(int(rms))

    for value in parameter_dict.values():
        row_container.append(value)

    #region testing export header rows
    with open(export_path_file, 'a',
              newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        writer.writerow(row_container)

    #endregion
    #endregion






selected_features = SetOfFeatures.all_features
epochs = 800
lr = 0.001
batch_size = 64
validation_split = 0.2
optimizer = "adam"
logarithm = False
normalize = False
batchnorm = False
exponent = False
train_split = 0.7
file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/Data/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
name = 'linear regression'
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
    'file_path': file_path
}
x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent, selected_features)

for i in range(3):
    # prediction = random_forrest(x_tr, y_tr, x_te, L2_matrix)
    prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
    write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

