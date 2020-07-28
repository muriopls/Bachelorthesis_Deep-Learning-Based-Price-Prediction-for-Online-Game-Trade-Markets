# region Handcrafting Dictionaries
positions = {
    'CAM': 32,
    'RW': 27,
    'CF': 23,
    'ST': 17,
    'LW': 13,
    'CM': 10,
    'CDM': 8,
    'RB': 6,
    'LF': 6,
    'LB': 5,
    'CB': 5,
    'LM': 4,
    'RF': 3,
    'RWB': 3,
    'LWB': 3,
    'RM': 3
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
    'Icons': 94,
    'LaLiga Santander': 11,
    'Ligue 1 Conforama': 11,
    'Premier League': 8,
    'Serie A TIM': 4,
    'Allsvenskan': 4,
    'Bundesliga': 2,
    'K LEAGUE Classic': 2,
    'CONMEBOL Libertadores': 2,
    'EFL League One': 2,
    'Liga Dimayor': 2,
    'Hyundai A-League': 2,
    'EFL League Two': 2,
    'Campeonato Scotiabank': 2,
    'Dominoâ€™s Ligue 2': 1,
    '3. Liga': 1,
    'Eredivisie': 1,
    'SAF': 1,
    'Calcio B': 1,
    'Ukraine Liga': 1,
    'CSL': 1,
    'Scottish Premiership': 1,
    'Raiffeisen Super League': 1,
    'EFL Championship': 1,
    'Ã–sterreichische FuÃŸball-Bundesliga': 1,
    'Croatia Liga': 1,
    'Saudi Professional League': 1,
    'Liga NOS': 1,
    'Superliga': 1,
    'Meiji Yasuda J1 League': 1,
    'League of Russia': 1,
    'LIGA Bancomer MX': 1,
    'SÃ¼per Lig': 1,
    'Bundesliga 2': 0,
    'ÄŒeskÃ¡ Liga': 0,
    'UAE Gulf League': 0,
    'Hellas Liga': 0,
    'Belgium Pro League': 0,
    'Major League Soccer': 0,
    'LaLiga 1 I 2 I 3': 0
}

nations = {
    'Bulgaria': 73,
    'Brazil': 29,
    'Egypt': 23,
    'France': 20,
    'Gabon': 17,
    'Ghana': 12,
    'Argentina': 11,
    'Finland': 10,
    'CÃ´te d\'Ivoire': 10,
    'England': 10,
    'Belgium': 9,
    'Czech Republic': 9,
    'Germany': 6,
    'Central African Republic': 4,
    'Eritrea': 4,
    'Denmark': 4,
    'Algeria': 4,
    'Burundi': 2,
    'Austria': 2,
    'Ecuador': 2,
    'Chad': 2,
    'Bosnia and Herzegovina': 2,
    'Croatia': 2,
    'Bermuda': 1,
    'Gambia': 1,
    'Canada': 1,
    'Australia': 1,
    'Colombia': 1,
    'Chile': 1,
    'Cameroon': 1,
    'China PR': 1,
    'Greece': 1,
    'FYR Macedonia': 1,
    'Costa Rica': 1,
    'Hungary': 1,
    'Iran': 1,
    'Angola': 1,
    'Congo DR': 1,
    'Armenia': 0,
    'Albania': 0,
    'Cape Verde Islands': 0,
    'Burkina Faso': 0,
    'Dominican Republic': 0,
    'Iceland': 0,
    'Guinea': 0,
    'Cyprus': 0,
    'Iraq': 0,
    'Georgia': 0,
    'Estonia': 0,
    'Israel': 0,
    'Honduras': 0,
    'Equatorial Guinea': 0,
    'Guinea-Bissau': 0,
    'Cuba': 0,
    'Benin': 0
}
# positions = {
#     'GK': 1,
#     'RB': 2,
#     'RWB': 2,
#     'CB': 3,
#     'LB': 2,
#     'LWB': 2,
#     'CDM': 5,
#     'CM': 5,
#     'CAM': 5,
#     'CF': 5,
#     'ST': 5,
#     'LM': 4,
#     'LF': 4,
#     'LW': 4,
#     'RM': 4,
#     'RF': 4,
#     'RW': 4,
# }
#
# leagues = {
#     'Icons': 10,
#     'Premier League': 4,
#     'LaLiga Santander': 4,
#     'Serie A TIM': 3,
#     'Ligue 1 Conforama': 3,
#     'Bundesliga': 2,
#     'Eridivisie': 2,
#     'Liga NOS': 2,
#     'NaN': 1
# }
#
# nations = {
#     'France': 5,
#     'Brazil': 5,
#     'England': 4,
#     'Spain': 4,
#     'Germany': 3,
#     'Italy': 3,
#     'Belgium': 3,
#     'Argentina': 3,
#     'Netherlands': 3,
#     'Portugal': 3,
#     'Croatia': 2,
#     'Colombia': 2,
#     'Poland': 2,
#     'Turkey': 2,
#     'NaN': 1
# }

working_rate = {
    'Low': 1,
    'Med': 2,
    'High': 3
}

strong_foot = {
    'Right': 1,
    'Left': 2
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