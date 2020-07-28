import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from dicts import positions, leagues, working_rate, nations, strong_foot


def get_features(data, selected_features):
    if selected_features == 0:
        features = data[['avg_price', 'overall_rating', 'skills', 'weakfoot']].copy()
    elif selected_features == 1:
        features = data[['avg_price', 'overall_rating', 'skills', 'weakfoot', 'defensive_work_rates', 'offensive_work_rates']].copy()
    elif selected_features == 2:
        features = data[['avg_price', 'overall_rating', 'skills', 'weakfoot', 'defensive_work_rates', 'offensive_work_rates', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 3:
        features = data[['avg_price', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 4:
        features = data[['avg_price', 'overall_rating', 'league', 'nation', 'weakfoot', 'pac', 'sho', 'pas', 'dri', 'def', 'phy']].copy()
    elif selected_features == 5:
        features = data[['avg_price', 'acceleration', 'sprintspeed', 'positioning', 'finishing', 'shotpower', 'longshots', 'volleys', 'penalties', 'vision', 'crossing', 'freekickaccuracy', 'shortpassing', 'longpassing', 'curve', 'agility', 'balance', 'reactions', 'ballcontrol', 'dribbling', 'composure', 'interceptions', 'headingaccuracy', 'marking', 'standingtackle', 'slidingtackle', 'jumping', 'stamina', 'strength', 'aggression']].copy()
    elif selected_features == 6:
        features = data.copy()
    elif selected_features == 7:
        features = data[['avg_price', 'overall_rating', 'league', 'position', 'weakfoot', 'acceleration', 'sprintspeed', 'agility', 'balance', 'composure', 'stamina', 'strength', ]].copy()
    elif selected_features == 8:
        features = data.copy()
        del features['defensive_work_rates']
        del features['offensive_work_rates']
        del features['origin']
        del features['birthday']
        del features['international_rep']
        del features['revision']
    elif selected_features == 9:
        features = data[['avg_price', 'overall_rating', 'acceleration', 'sprintspeed', 'positioning', 'finishing', 'shotpower', 'longshots', 'volleys', 'penalties', 'vision', 'crossing', 'freekickaccuracy', 'shortpassing', 'longpassing', 'curve', 'agility', 'balance', 'reactions', 'ballcontrol', 'dribbling', 'composure', 'interceptions', 'headingaccuracy', 'marking', 'standingtackle', 'slidingtackle', 'jumping', 'stamina', 'strength', 'aggression']].copy()


    return features


def handcraft_features(data_to_handcraft):
    data_to_handcraft['position'] = data_to_handcraft['position'].map(positions)
    data_to_handcraft['league'] = data_to_handcraft['league'].map(leagues)
    data_to_handcraft['league'] = data_to_handcraft['league'].fillna(0)
    data_to_handcraft['nation'] = data_to_handcraft['nation'].map(nations)
    data_to_handcraft['nation'] = data_to_handcraft['nation'].fillna(0)
    data_to_handcraft['defensive_work_rates'] = data_to_handcraft['defensive_work_rates'].map(working_rate)
    data_to_handcraft['offensive_work_rates'] = data_to_handcraft['offensive_work_rates'].map(working_rate)
    data_to_handcraft['foot'] = data_to_handcraft['foot'].map(strong_foot)
    return data_to_handcraft


def import_data(log, norm, file_path, train_split, selected_features):
    # region Import data
    imported_data = pd.read_csv(file_path, sep=';', encoding='mac_roman')
    target_scaler = None
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

    features = get_features(imported_data, selected_features)

    # features['overall_rating'] = pd.to_numeric(features['overall_rating'], errors='coerce')
    # features['pac'] = pd.to_numeric(features['pac'], errors='coerce')
    # features['sho'] = pd.to_numeric(features['sho'], errors='coerce')
    # features['pas'] = pd.to_numeric(features['pas'], errors='coerce')
    # features['dri'] = pd.to_numeric(features['dri'], errors='coerce')
    # features['def'] = pd.to_numeric(features['def'], errors='coerce')
    # features['phy'] = pd.to_numeric(features['phy'], errors='coerce')
    # features['acceleration'] = pd.to_numeric(features['acceleration'], errors='coerce')
    # features['sprintspeed'] = pd.to_numeric(features['sprintspeed'], errors='coerce')
    # features['positioning'] = pd.to_numeric(features['positioning'], errors='coerce')
    # features['finishing'] = pd.to_numeric(features['finishing'], errors='coerce')
    # features['shotpower'] = pd.to_numeric(features['shotpower'], errors='coerce')
    # features['longshots'] = pd.to_numeric(features['longshots'], errors='coerce')
    # features['volleys'] = pd.to_numeric(features['volleys'], errors='coerce')
    # features['penalties'] = pd.to_numeric(features['penalties'], errors='coerce')
    # features['vision'] = pd.to_numeric(features['vision'], errors='coerce')
    # features['crossing'] = pd.to_numeric(features['crossing'], errors='coerce')
    # features['freekickaccuracy'] = pd.to_numeric(features['freekickaccuracy'], errors='coerce')
    # features['shortpassing'] = pd.to_numeric(features['shortpassing'], errors='coerce')
    # features['longpassing'] = pd.to_numeric(features['longpassing'], errors='coerce')
    # features['curve'] = pd.to_numeric(features['curve'], errors='coerce')
    # features['agility'] = pd.to_numeric(features['agility'], errors='coerce')
    # features['balance'] = pd.to_numeric(features['balance'], errors='coerce')
    # features['reactions'] = pd.to_numeric(features['reactions'], errors='coerce')
    # features['ballcontrol'] = pd.to_numeric(features['ballcontrol'], errors='coerce')
    # features['dribbling'] = pd.to_numeric(features['dribbling'], errors='coerce')
    # features['composure'] = pd.to_numeric(features['composure'], errors='coerce')
    # features['interceptions'] = pd.to_numeric(features['interceptions'], errors='coerce')
    # features['headingaccuracy'] = pd.to_numeric(features['headingaccuracy'], errors='coerce')
    # features['marking'] = pd.to_numeric(features['marking'], errors='coerce')
    # features['standingtackle'] = pd.to_numeric(features['standingtackle'], errors='coerce')
    # features['slidingtackle'] = pd.to_numeric(features['slidingtackle'], errors='coerce')
    # features['jumping'] = pd.to_numeric(features['jumping'], errors='coerce')
    # features['stamina'] = pd.to_numeric(features['stamina'], errors='coerce')
    # features['strength'] = pd.to_numeric(features['strength'], errors='coerce')
    # features['aggression'] = pd.to_numeric(features['aggression'], errors='coerce')
    # endregion

    # region train and test split

    L2 = np.sqrt(np.sum(np.multiply(features, features), axis=0))

    x_data = features.iloc[:, 1:]
    y_data = features.iloc[:, :1]

    x_data = tf.keras.utils.normalize(x_data)

    # Logarithm
    if norm:
        #target_scaler = MinMaxScaler()
        #target_scaler.fit(y_data)
        #y_data = pd.DataFrame(target_scaler.transform(y_data))

        y_data = y_data/L2[0]
    # Normalization
    if log:
        y_data = np.log10(y_data)

    number_of_train_rows = int(len(features.index) * train_split)

    x_train = x_data.iloc[:number_of_train_rows, :]
    y_train = y_data.iloc[:number_of_train_rows, :]
    x_test = x_data.iloc[number_of_train_rows:, :]
    y_test = y_data.iloc[number_of_train_rows:, :]
    # endregion

    # endregion
    return x_train, y_train, x_test, y_test, L2, saved_dataframe, target_scaler
