import csv
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import keras
import keras.layers as layers
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import sqrt
from keras import losses
from keras.layers import Dropout
from pathlib import Path
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor



from tabulate import tabulate


# region Predefining
class SetOfFeatures:
    minimum = 0
    minimum_n_wr = 1
    minimum_n_base_attributes = 2
    only_base_attributes = 3
    base_n_handcrafted = 4
    only_detail_attributes = 5
    all_features = 6
    custom_1 = 7
    custom_2 = 8

selected_features = SetOfFeatures.all_features
price_difference_path = 'C:/Users/murio/PycharmProjects/pricePrediction/price_differences'
epochs = 150
lr = 0.001
batch_size = 32
validation_split = 0.3
logarithm = False
normalize = False
batchnorm = False


# region Handcrafting Dictionaries
positions = {
    'GK': 1,
    'RB': 2,
    'RWB': 2,
    'CB': 3,
    'LB': 2,
    'LWB': 2,
    'CDM': 5,
    'CM': 5,
    'CAM': 5,
    'CF': 5,
    'ST': 5,
    'LM': 4,
    'LF': 4,
    'LW': 4,
    'RM': 4,
    'RF': 4,
    'RW': 4,
}

working_rate = {
    'Low': 1,
    'Med': 2,
    'High': 3
}

strong_foot = {
    'Right': 1,
    'Left': 2
}

leagues = {
    'Icons': 10,
    'Premier League': 4,
    'LaLiga Santander': 4,
    'Serie A TIM': 3,
    'Ligue 1 Conforama': 3,
    'Bundesliga': 2,
    'Eridivisie': 2,
    'Liga NOS': 2,
    'NaN': 1
}

nations = {
    'France': 5,
    'Brazil': 5,
    'England': 4,
    'Spain': 4,
    'Germany': 3,
    'Italy': 3,
    'Belgium': 3,
    'Argentina': 3,
    'Netherlands': 3,
    'Portugal': 3,
    'Croatia': 2,
    'Colombia': 2,
    'Poland': 2,
    'Turkey': 2,
    'NaN': 1
}
#endregion


# region methods
def get_features(data):
    if selected_features == 0:
        features = data[['avg price', 'overall_rating', 'skills', 'weakfoot']].copy()
    elif selected_features == 1:
        features = data[['avg price', 'overall_rating', 'skills', 'weakfoot', 'defensive_work_rates', 'offensive_work_rates']].copy()
    elif selected_features == 2:
        features = data[['avg price', 'overall_rating', 'skills', 'weakfoot', 'defensive_work_rates', 'offensive_work_rates', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 3:
        features = data[['avg price', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 4:
        features = data[['avg price', 'overall_rating', 'league', 'nation', 'weakfoot', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 5:
        features = data[['avg price', 'acceleration', 'sprintspeed', 'positioning', 'finishing', 'shotpower', 'longshots', 'volleys', 'penalties', 'vision', 'crossing', 'freekickaccuracy', 'shortpassing', 'longpassing', 'curve', 'agility', 'balance', 'reactions', 'ballcontrol', 'dribbling', 'composure', 'interceptions', 'headingaccuracy', 'marking', 'standingtackle', 'slidingtackle', 'jumping', 'stamina', 'strength', 'aggression']].copy()
    elif selected_features == 6:
        features = data.copy()
    elif selected_features == 7:
        features = data[['avg price', 'overall_rating', 'league', 'position', 'weakfoot', 'acceleration', 'sprintspeed', 'agility', 'balance', 'composure', 'stamina', 'strength', ]].copy()
    elif selected_features == 8:
        features = data.copy()
        del features['defensive_work_rates']
        del features['offensive_work_rates']
        del features['origin']
        del features['birthday']
        del features['international_rep']
        del features['revision']

    return features


def train_and_evaluate_models(batchnorm, x_train, y_train, x_test, L2, optimizer, test_name, to_disk):
    model = keras.models.Sequential()



    # region create and train model
    print('create neural network')
    model.add(layers.Dense(512, use_bias=False))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(256, use_bias=False))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(128, use_bias=False))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, use_bias=False))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(32, use_bias=False))
    model.add(layers.Dense(1, use_bias=False))

    log_dir = ""

    to_disk = False
    if to_disk:
        log_dir = "E:\Bachelorarbeit_Informatik\Auswertung\logs\{}\{}_{}".format(experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"),test_name)
        # os.mkdir("E:\Bachelorarbeit_Informatik\Auswertung\logs\{}".format(experiment_time))
        # os.mkdir("E:\Bachelorarbeit_Informatik\Auswertung\logs\{}\{}_{}".format(experiment_time,
        #                                                                     datetime.now().strftime("%d%m%Y_%H%M%S"),
        #                                                                     test_name))
    else:
        log_dir = "logs\\{}\\{}_{}".format(experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name)
        # os.mkdir("logs\\{}".format(experiment_time))
        # os.mkdir("logs\\{}\\{}_{}".format(experiment_time, datetime.now().strftime("%d%m%Y_%H%M%S"), test_name))

    print(log_dir)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir= log_dir,
        histogram_freq=1,
        write_graph=True, write_images=True)

    model.compile(optimizer=optimizer(lr=lr),
                  loss='mean_squared_error')

    print('train neural network')
    history = model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks = [tensorboard])
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
    csv_header.append('Difference Relative')
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
            feature_list.append(int(y_test_values.iloc[i].values[0])-int(prediction_list[i]))
            feature_list.append(int(y_test_values.iloc[i].values[0]))
            feature_list.append(int(prediction_list[i]))
            # If predicted value == 0
            if int(prediction_list[i]) != 0:
                feature_list.append(float(int(y_test_values.iloc[i].values[0])/int(prediction_list[i])))
            else:
                feature_list.append((int(y_test_values.iloc[i].values[0]), int(prediction_list[i]), 'error'))
            feature_list.append((saved_dataframe.get_value(index_list[i], 'name')))

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


def handcraft_features(data_to_handcraft):
    data_to_handcraft['position'] = data_to_handcraft['position'].map(positions)
    data_to_handcraft['league'] = data_to_handcraft['league'].map(leagues)
    data_to_handcraft['league'] = data_to_handcraft['league'].fillna(0)
    data_to_handcraft['nation'] = data_to_handcraft['nation'].map(nations)
    data_to_handcraft['nation'] = data_to_handcraft['nation'].fillna(1)
    data_to_handcraft['defensive_work_rates'] = data_to_handcraft['defensive_work_rates'].map(working_rate)
    data_to_handcraft['offensive_work_rates'] = data_to_handcraft['offensive_work_rates'].map(working_rate)
    data_to_handcraft['foot'] = data_to_handcraft['foot'].map(strong_foot)
    return data_to_handcraft

# endregion


# endregion

def import_data(log, norm, file_path, train_split, exponent):
    # region Import data
    imported_data = pd.read_csv(file_path, sep=';', encoding='mac_roman')

    # endregion

    imported_data = handcraft_features(imported_data)
    # data = data.sample(frac=1)

    # region Preprocess data (categorization of non numeric values)
    saved_dataframe = imported_data.copy()

    imported_data['name'] = pd.Categorical(imported_data['name'], ordered=False)
    names = imported_data.name.cat
    imported_data['name'] = imported_data.name.cat.codes
    imported_data['club'] = pd.Categorical(imported_data['club'])
    imported_data['club'] = imported_data.club.cat.codes
    # data['nation'] = pd.Categorical(data['nation'])
    # data['nation'] = data.nation.cat.codes
    # data['league'] = pd.Categorical(data['league'])
    # data['league'] = data.league.cat.codes
    # data['position'] = pd.Categorical(data['position'])
    # data['position'] = data.position.cat.codes
    # data['foot'] = pd.Categorical(data['foot'])
    # data['foot'] = data.foot.cat.codes
    imported_data['height'] = pd.Categorical(imported_data['height'])
    imported_data['height'] = imported_data.height.cat.codes
    imported_data['revision'] = pd.Categorical(imported_data['revision'])
    imported_data['revision'] = imported_data.revision.cat.codes
    imported_data['origin'] = pd.Categorical(imported_data['origin'])
    imported_data['origin'] = imported_data.origin.cat.codes
    imported_data['birthday'] = pd.Categorical(imported_data['birthday'])
    imported_data['birthday'] = imported_data.birthday.cat.codes
    imported_data['international_rep'] = pd.Categorical(imported_data['international_rep'])
    imported_data['international_rep'] = imported_data.international_rep.cat.codes

    features = get_features(imported_data)
    print(features.dtypes)
    features['overall_rating'] = pd.to_numeric(features['overall_rating'], errors='coerce')
    features['pac'] = pd.to_numeric(features['pac'], errors='coerce')
    features['sho'] = pd.to_numeric(features['sho'], errors='coerce')
    features['pas'] = pd.to_numeric(features['pas'], errors='coerce')
    features['dri'] = pd.to_numeric(features['dri'], errors='coerce')
    features['def'] = pd.to_numeric(features['def'], errors='coerce')
    features['phy'] = pd.to_numeric(features['phy'], errors='coerce')
    features['acceleration'] = pd.to_numeric(features['acceleration'], errors='coerce')
    features['sprintspeed'] = pd.to_numeric(features['sprintspeed'], errors='coerce')
    features['positioning'] = pd.to_numeric(features['positioning'], errors='coerce')
    features['finishing'] = pd.to_numeric(features['finishing'], errors='coerce')
    features['shotpower'] = pd.to_numeric(features['shotpower'], errors='coerce')
    features['longshots'] = pd.to_numeric(features['longshots'], errors='coerce')
    features['volleys'] = pd.to_numeric(features['volleys'], errors='coerce')
    features['penalties'] = pd.to_numeric(features['penalties'], errors='coerce')
    features['vision'] = pd.to_numeric(features['vision'], errors='coerce')
    features['crossing'] = pd.to_numeric(features['crossing'], errors='coerce')
    features['freekickaccuracy'] = pd.to_numeric(features['freekickaccuracy'], errors='coerce')
    features['shortpassing'] = pd.to_numeric(features['shortpassing'], errors='coerce')
    features['longpassing'] = pd.to_numeric(features['longpassing'], errors='coerce')
    features['curve'] = pd.to_numeric(features['curve'], errors='coerce')
    features['agility'] = pd.to_numeric(features['agility'], errors='coerce')
    features['balance'] = pd.to_numeric(features['balance'], errors='coerce')
    features['reactions'] = pd.to_numeric(features['reactions'], errors='coerce')
    features['ballcontrol'] = pd.to_numeric(features['ballcontrol'], errors='coerce')
    features['dribbling'] = pd.to_numeric(features['dribbling'], errors='coerce')
    features['composure'] = pd.to_numeric(features['composure'], errors='coerce')
    features['interceptions'] = pd.to_numeric(features['interceptions'], errors='coerce')
    features['headingaccuracy'] = pd.to_numeric(features['headingaccuracy'], errors='coerce')
    features['marking'] = pd.to_numeric(features['marking'], errors='coerce')
    features['standingtackle'] = pd.to_numeric(features['standingtackle'], errors='coerce')
    features['slidingtackle'] = pd.to_numeric(features['slidingtackle'], errors='coerce')
    features['jumping'] = pd.to_numeric(features['jumping'], errors='coerce')
    features['stamina'] = pd.to_numeric(features['stamina'], errors='coerce')
    features['strength'] = pd.to_numeric(features['strength'], errors='coerce')
    features['aggression'] = pd.to_numeric(features['aggression'], errors='coerce')
    print(features.dtypes)

    # endregion

    if exponent:
        print(features['overall_rating'])
        features['overall_rating'] = pd.to_numeric(features['overall_rating'], errors='coerce') ** 10
        print(features['overall_rating'])
        features['pac'] = pd.to_numeric(features['pac'], errors='coerce') ** 10
        features['sho'] = pd.to_numeric(features['sho'], errors='coerce') ** 10
        features['pas'] = pd.to_numeric(features['pas'], errors='coerce') ** 10
        features['dri'] = pd.to_numeric(features['dri'], errors='coerce') ** 10
        features['def'] = pd.to_numeric(features['def'], errors='coerce') ** 10
        features['phy'] = pd.to_numeric(features['phy'], errors='coerce') ** 10
        features['acceleration'] = pd.to_numeric(features['acceleration'], errors='coerce') ** 10
        features['sprintspeed'] = pd.to_numeric(features['sprintspeed'], errors='coerce') ** 10
        features['positioning'] = pd.to_numeric(features['positioning'], errors='coerce') ** 10
        features['finishing'] = pd.to_numeric(features['finishing'], errors='coerce') ** 10
        features['shotpower'] = pd.to_numeric(features['shotpower'], errors='coerce') ** 10
        features['longshots'] = pd.to_numeric(features['longshots'], errors='coerce') ** 10
        features['volleys'] = pd.to_numeric(features['volleys'], errors='coerce') ** 10
        features['penalties'] = pd.to_numeric(features['penalties'], errors='coerce') ** 10
        features['vision'] = pd.to_numeric(features['vision'], errors='coerce') ** 10
        features['crossing'] = pd.to_numeric(features['crossing'], errors='coerce') ** 10
        features['freekickaccuracy'] = pd.to_numeric(features['freekickaccuracy'], errors='coerce') ** 10
        features['shortpassing'] = pd.to_numeric(features['shortpassing'], errors='coerce') ** 10
        features['longpassing'] = pd.to_numeric(features['longpassing'], errors='coerce') ** 10
        features['curve'] = pd.to_numeric(features['curve'], errors='coerce') ** 10
        features['agility'] = pd.to_numeric(features['agility'], errors='coerce') ** 10
        features['balance'] = pd.to_numeric(features['balance'], errors='coerce') ** 10
        features['reactions'] = pd.to_numeric(features['reactions'], errors='coerce') ** 10
        features['ballcontrol'] = pd.to_numeric(features['ballcontrol'], errors='coerce') ** 10
        features['dribbling'] = pd.to_numeric(features['dribbling'], errors='coerce') ** 10
        features['composure'] = pd.to_numeric(features['composure'], errors='coerce') ** 10
        features['interceptions'] = pd.to_numeric(features['interceptions'], errors='coerce') ** 10
        features['headingaccuracy'] = pd.to_numeric(features['headingaccuracy'], errors='coerce') ** 10
        features['marking'] = pd.to_numeric(features['marking'], errors='coerce') ** 10
        features['standingtackle'] = pd.to_numeric(features['standingtackle'], errors='coerce') ** 10
        features['slidingtackle'] = pd.to_numeric(features['slidingtackle'], errors='coerce') ** 10
        features['jumping'] = pd.to_numeric(features['jumping'], errors='coerce') ** 10
        features['stamina'] = pd.to_numeric(features['stamina'], errors='coerce') ** 10
        features['strength'] = pd.to_numeric(features['strength'], errors='coerce') ** 10
        features['aggression'] = pd.to_numeric(features['aggression'], errors='coerce') ** 10

    # region train and test split

    L2 = np.sqrt(np.sum(np.multiply(features, features), axis=0))

    x_data = features.iloc[:, 1:]
    y_data = features.iloc[:, :1]

    x_data = tf.keras.utils.normalize(x_data)

    print(y_data.iloc[0])
    # Logarithm
    if norm:
        y_data = y_data/L2[0]
    # Normalization
    if log:
        y_data = np.log10(y_data)
    print(y_data.iloc[0])
    print(L2)
    print(y_data.iloc[0]*L2[0])

    number_of_train_rows = int(len(features.index) * train_split)

    x_train = x_data.iloc[:number_of_train_rows, :]
    y_train = y_data.iloc[:number_of_train_rows, :]
    x_test = x_data.iloc[number_of_train_rows:, :]
    y_test = y_data.iloc[number_of_train_rows:, :]

    # x_train = x_train.to_numpy()
    # y_train = y_train.to_numpy()
    # x_test = x_test.to_numpy()
    # y_test = y_test.to_numpy()




    # endregion

    # endregion
    return x_train, y_train, x_test, y_test, L2, saved_dataframe


experiment_time = datetime.now().strftime("%Y%m%d_%H%M")
export_to_disk = False

big_export_path_file = ""
export_path_file = ""

if export_to_disk:
    big_export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "E:\Bachelorarbeit_Informatik\Auswertung/{}_{}.csv".format(experiment_time, "rmse")
else:
    big_export_path_file = "C:/Users/murio/PycharmProjects/pricePrediction/price_differences/{}_{}.csv".format(
        experiment_time, "price_differences")
    export_path_file = "C:/Users/murio/PycharmProjects/pricePrediction/optimization/{}_{}.csv".format(experiment_time, "rmse")

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

selected_features = SetOfFeatures.all_features
epochs = 150
lr = 0.01
batch_size = 128
validation_split = 0.3
optimizer = RMSprop
logarithm = False
normalize = False
batchnorm = False
exponent = False
train_split = 0.7
file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
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
x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)

for i in range(3):
    prediction = random_forrest(x_tr, y_tr, x_te, L2_matrix)
    # prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
    write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)

# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.1
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = False
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.1'
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
#     'file_path': file_path
# }
#
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#
# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.05
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = True
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.05'
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
#     'file_path': file_path
# }
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#
# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = False
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.01'
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
#     'file_path': file_path
# }
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#
#
# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.005
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = False
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.005'
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
#     'file_path': file_path
# }
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#
# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.001
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = False
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.001'
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
#     'file_path': file_path
# }
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#
# selected_features = SetOfFeatures.all_features
# epochs = 100
# lr = 0.0001
# batch_size = 32
# validation_split = 0.3
# optimizer = RMSprop
# logarithm = False
# normalize = False
# batchnorm = False
# exponent = False
# train_split = 0.7
# file_path = 'C:/Users/murio/PycharmProjects/pricePrediction/first_approach/PriceSnapshot_22-04-2020_xbox_modified.csv'
# name = 'lr_0.0001'
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
#     'file_path': file_path
# }
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path, train_split, exponent)
# for i in range(2):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, optimizer, name, export_to_disk)
#     write_price_differences(prediction, x_te, y_te, L2_matrix, saved_data, parameter_dict)
#




# print(svc_param_selection(x_tr, y_tr))


# print("SVM now")
# selected_features = SetOfFeatures.all_features
# logarithm = False
# normalize = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize, file_path)
# print("Imported Data")
# prediction = support_vector_machine(x_tr, y_tr, x_te, L2_matrix)
# print("Training done")
# write_price_differences(prediction, name, x_te, y_te, L2_matrix, saved_data)

# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = True
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'logarithmed y values')
#     write_price_differences(prediction, 'normalized y values', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = True
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'logarithmed y values')
#     write_price_differences(prediction, 'logarithmed y values', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'initial y values')
#     write_price_differences(prediction, 'initial y values', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = True
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'batch normalization')
#     write_price_differences(prediction, 'batch normalization', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, Adam, 'adam optimizer')
#     write_price_differences(prediction, 'adam optimizer', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.1
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'learning rate = 0.1')
#     write_price_differences(prediction, 'learning rate = 0.1', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.05
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'learning rate = 0.05')
#     write_price_differences(prediction, 'learning rate = 0.05', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'learning rate = 0.01')
#     write_price_differences(prediction, 'learning rate = 0.01', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.005
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'learning rate = 0.005')
#     write_price_differences(prediction, 'learning rate = 0.005', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'learning rate = 0.001')
#     write_price_differences(prediction, 'learning rate = 0.001', x_te, y_te, L2_matrix, saved_data)
#
#
# epochs = 50
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, '50 Epochs')
#     write_price_differences(prediction, '50 Epochs', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 100
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, '100 Epochs')
#     write_price_differences(prediction, '100 Epochs', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 250
# lr = 0.01
# batch_size = 32
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, '250 Epochs')
#     write_price_differences(prediction, '250 Epochs', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 64
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'batchsize 64')
#     write_price_differences(prediction, 'batchsize 64', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 128
# validation_split = 0.3
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'batchsize 128')
#     write_price_differences(prediction, 'batchsize 128', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.2
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'validation split = 0.2')
#     write_price_differences(prediction, 'validation split = 0.2', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.4
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'validation split = 0.4')
#     write_price_differences(prediction, 'validation split = 0.4', x_te, y_te, L2_matrix, saved_data)
#
# epochs = 150
# lr = 0.01
# batch_size = 32
# validation_split = 0.5
# logarithm = False
# normalize = False
# batchnorm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop, 'validation split = 0.5')
#     write_price_differences(prediction, 'validation split = 0.5', x_te, y_te, L2_matrix, saved_data)

# validation_split = 0.3
# batchnorm = True
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(10):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop)
#     write_price_differences(prediction, '0.3', x_te, y_te, L2_matrix, saved_data)

# validation_split = 0.4
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(10):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop)
#     write_price_differences(prediction, '0.4', x_te, y_te, L2_matrix, saved_data)
#
# validation_split = 0.5
# #logarithm = True
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(10):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop)
#     write_price_differences(prediction, '0.5', x_te, y_te, L2_matrix, saved_data)

# epochs=500
# #logarithm = True
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix, RMSprop)
#     write_price_differences(prediction, '500', x_te, y_te, L2_matrix, saved_data)


# normalize = True
# logarithm = False
# x_tr, y_tr, x_te, y_te, L2_matrix, saved_data = import_data(logarithm, normalize)
# for i in range(20):
#     prediction = train_and_evaluate_models(batchnorm, x_tr, y_tr, x_te, L2_matrix)
#     write_price_differences(prediction, 'Normalized y values', x_te, y_te, L2_matrix, saved_data)

#batchnorm = False


