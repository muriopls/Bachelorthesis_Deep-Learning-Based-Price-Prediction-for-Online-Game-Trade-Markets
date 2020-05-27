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