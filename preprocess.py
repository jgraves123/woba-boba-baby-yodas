import numpy as np
import csv


def get_data(data_file, all_cat):
    """
    Takes in csv file and reads it to gather training and testing data and labels.
    :param data_file: file path of the csv data
    :return: data from relevant columns of csv as np array, tuple of (woba values, events mapped as int IDs)
    """
    data_dict = {}
    # column titles in the CSV we want to use
    columns_we_want = np.array(['pitch_type', 'batter', 'player_name', 'events', 'stand', 'p_throws', 'home_team',
                                'away_team', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning',
                                'inning_topbot', 'woba_value', 'bat_score', 'fld_score', 'if_fielding_alignment',
                                'of_fielding_alignment'])

    # initialize empty lists in dictionary to add to, where keys are column titles
    for col in columns_we_want:
        data_dict[col] = []
    # read the CSV file and append each piece of data that we want from relevant columns to dictionary's relevant value
    # list
    csv_reader = csv.DictReader(open(data_file))
    for row in csv_reader:
        for col in columns_we_want:
            data_dict[col].append(row[col])
    # convert value lists to numpy arrays to manipulate them later
    for col in columns_we_want:
        data_dict[col] = np.array(data_dict[col])

    print("Done creating columns...")

    data_dict['woba_value'] = data_dict['woba_value'].astype(float)

    # build IDs maps string data in certain columns to IDs using build_ids function
    data_dict['pitch_type'] = build_ids(data_dict['pitch_type'])
    data_dict['batter'], batter_dict, batter_woba = batter_pitcher_woba(data_dict['batter'], data_dict['woba_value'])
    data_dict['player_name'], pitcher_dict, pitcher_woba = batter_pitcher_woba(data_dict['player_name'], data_dict['woba_value'])
    data_dict['if_fielding_alignment'] = build_ids(data_dict['if_fielding_alignment'])
    data_dict['of_fielding_alignment'] = build_ids(data_dict['of_fielding_alignment'])
    
    # we not only assign an int ID to each possible event, but we also assosciate individual
    # events with specific int IDs for reference as labels in our loss function
    data_dict['events'], labels_dictionary, woba_array = build_labels(data_dict['events'], data_dict['woba_value'], all_cat)

    # 1 = player on a given base, 0 = nobody on the base
    data_dict['on_3b'] = np.where(data_dict['on_3b'] == 'null', 0, 1)
    data_dict['on_2b'] = np.where(data_dict['on_2b'] == 'null', 0, 1)
    data_dict['on_1b'] = np.where(data_dict['on_1b'] == 'null', 0, 1)
    # batter stance and pitcher handedness encoded as L = 0, R = 1
    data_dict['stand'] = np.where(data_dict['stand'] == 'L', 0, 1)
    data_dict['p_throws'] = np.where(data_dict['p_throws'] == 'L', 0, 1)

    # get the fielding and hitting team names
    data_dict['away_team'], data_dict['home_team'] = field_team(data_dict['inning_topbot'], data_dict['home_team'],
                                                                data_dict['away_team'])

    # away_team key now represents the fielding team name, which we now map to IDs
    data_dict['away_team'], team_dict = build_ids_dict(data_dict['away_team'])

    # bat_score column now represents the difference between scores of hitting and fielding team
    data_dict['bat_score'] = data_dict['bat_score'].astype(np.int32) - data_dict['fld_score'].astype(np.int32)

    # stack all the data to one massive array
    data_whole = np.column_stack((data_dict['pitch_type'], data_dict['batter'], data_dict['player_name'],
                                  data_dict['events'], data_dict['stand'], data_dict['p_throws'],
                                  data_dict['away_team'], data_dict['balls'], data_dict['strikes'], data_dict['on_3b'],
                                  data_dict['on_2b'], data_dict['on_1b'], data_dict['outs_when_up'],
                                  data_dict['inning'], data_dict['woba_value'], data_dict['bat_score'],
                                  data_dict['if_fielding_alignment'], data_dict['of_fielding_alignment']))

    print("Done column stacking...")

    # get rows that have null values that can't be encoded/ would lead to model confusion
    # pitch_types (column 0), infield shift (column 19), and outfield_shift (column 20) all have entries will null
    # data that we cannot use/encode to be meaningful data
    rows_to_delete = []
    for row_num in range(data_whole.shape[0]):
        if data_whole[row_num][0] == 'null' or data_whole[row_num][16] == 'null' or data_whole[row_num][17] == 'null':
            rows_to_delete.append(row_num)

    # delete rows that have null values we want to remove
    data_minus_nulls = np.delete(data_whole, rows_to_delete, axis=0)

    print("Done deleting null rows...")

    # separate out labels from data and remove the labels from the data array
    labels = (data_minus_nulls[:, 14].astype(np.float32), data_minus_nulls[:, 3].astype(np.float32))
    labels = np.column_stack(labels)
    relevant_data = np.delete(data_minus_nulls, [3, 14], axis=1).astype(np.int32)
    # remove labels and home_team, fld_score, and inning_topbot from columns we want
    columns_we_want = np.delete(columns_we_want, [3, 6, 15, 16, 18], axis=0)

    # index dict is a dictionary between column name and index number of column (ex: pitch type is column 0, batter is
    # column 1, etc...)
    index_dict = {}
    for i, e in enumerate(columns_we_want):
        index_dict[e] = i

    print("Done creating index dictionary...")

    # max dict is a dictionary between column name and number of unique values used in assignment.py to get number of
    # pitchers, batters, etc.
    max_dict = {}
    for i, e in enumerate(columns_we_want):
        max_dict[e] = np.amax(relevant_data[:, i]) + 1

    print("Done creating max value dictionary...")

    # we will now shuffle our data for SGD
    shuffle = np.arange(relevant_data.shape[0])
    np.random.shuffle(shuffle)
    shuffled_data = np.take(relevant_data, shuffle, axis=0)
    shuffled_labels = np.take(labels, shuffle, axis=0)

    print("Done shuffling data...")

    # split into training and testing data. 90% of data is training, remaining 10% is for testing
    data_training = shuffled_data[0:int(shuffled_data.shape[0]*0.9), :]
    data_testing = shuffled_data[int(shuffled_data.shape[0]*0.9):, :]
    labels_training = shuffled_labels[0:int(shuffled_labels.shape[0]*0.9), :]
    labels_testing = shuffled_labels[int(shuffled_labels.shape[0]*0.9):, :]

    batter_woba = woba_for_player(data_training, labels_training, index_dict['batter'])
    pitcher_woba = woba_for_player(data_training, labels_training, index_dict['player_name'])


# TODO delete labels_dict

    print("Done splitting data into training/testing with 90/10 split...")
    print("Done preprocessing!")
    return data_training, data_testing, labels_training, labels_testing, labels_dictionary, woba_array, \
        index_dict, max_dict, pitcher_dict, batter_dict, team_dict, batter_woba, pitcher_woba


def build_ids(column_data):
    """
    Takes in a column of data and changes every string ID to an int associated ID
    :param column_data: data_dict[key], a column of the data_dict used in get_data
    :return: Column of data now represented as ints instead of strings
    """
    # find all the unique string values within the column
    unique_values_list = np.unique(column_data)
    counter = 0
    for option in unique_values_list:
        # we don't want to remove null values since we need to know they're there in order to delete the whole row
        # of data within get_data. Any null values we want to have removed (like on_3b, on_2b, on_1b) are done in
        # get_data using np.where
        if option != "null":
            # associate value with an int
            column_data = np.where(column_data == option, counter, column_data)
            counter += 1
    return column_data


def field_team(top_bot, home, away):
    """
    Takes in what part of the inning it is and converts home and away team to pitching and hitting team
    :param top_bot: data_dict column array reprenting if it's top or bottom of the inning
    :param home: data_dict column array representing the home team's string name
    :param away: data_dict column array representing the away team's string name
    :return: array representing the fielding teams' string name, array representing the hitting teams' string name
    """
    # if it is the top of the inning, the away team is batting and home team is pitching
    field = np.where(top_bot == 'Top', home, away)
    hit = np.where(top_bot == 'Bot', away, home)
    return field, hit


def build_ids_dict(labels_col_data):
    """
    Takes in a column of labels that has the different types of events that can happen as a form of string names
    :param labels_col_data: data_dict[key], a column of the data_dict used in get_data
    :return: Column of data now represented as ints instead of strings, and the dictionary mapping string to the number
    """
    # find all the unique string values within the column
    labels_dict = {}
    unique_values_list = np.unique(labels_col_data)
    counter = 0
    for option in unique_values_list:
        # we don't want to remove null values since we need to know they're there in order to delete the whole row
        # of data within get_data. Any null values we want to have removed (like on_3b, on_2b, on_1b) are done in
        # get_data using np.where
        if option != "null":
            labels_dict[option] = counter
            # associate value with an int
            labels_col_data = np.where(labels_col_data == option, counter, labels_col_data)
            counter += 1
    return labels_col_data, labels_dict


def batter_pitcher_woba(labels_col_data, woba_col):
    """
        Takes in a column of labels that has the different types of events that can happen as a form of string names
        :param labels_col_data: data_dict[key], a column of the data_dict used in get_data
        :return: Column of data now represented as ints instead of strings, and the dictionary mapping string to the number
        """
    # find all the unique string values within the column
    avg_woba_dict = {}
    labels_dict = {}
    unique_values_list = np.unique(labels_col_data)
    counter = 0
    for option in unique_values_list:
        # we don't want to remove null values since we need to know they're there in order to delete the whole row
        # of data within get_data. Any null values we want to have removed (like on_3b, on_2b, on_1b) are done in
        # get_data using np.where
        if option != "null":
            labels_dict[option] = counter
            # associate value with an int
            player_values = woba_col[np.where(labels_col_data == option)]
            avg_woba_dict[counter] = np.mean(player_values)
            labels_col_data = np.where(labels_col_data == option, counter, labels_col_data)
            counter += 1
    return labels_col_data, labels_dict, avg_woba_dict


def woba_for_player(train_data, train_labels, index):
    avg_woba_dict = {}
    unique_values_list = np.unique(train_data[:, index])
    for option in unique_values_list:
        player_values = train_labels[:, 0][np.where(train_data[:, index] == option)]
        avg_woba_dict[option] = np.mean(player_values)
    return avg_woba_dict



def build_labels(labels_col_data, woba_column, all_cat):
    """
    Takes in a column of labels that has the different types of events that can happen as a form of string names
    :param woba_column: an array from data_dict with the woba value corresponding to every play outcome type
    :param labels_col_data: data_dict[key], a column of the data_dict used in get_data
    :return: Column of data now represented as ints instead of strings, and the dictionary mapping string to the number
    """
    # find all the unique string values within the column
    labels_dict = {}
    if all_cat:
        woba_array = [None]*19
    else:
        woba_array = [None]*6
    counter = 0

    if all_cat:
        labels_col_copy = np.copy(labels_col_data)
        for i, e in enumerate(labels_col_data):
            # we don't want to remove null values since we need to know they're there in order to delete the whole row
            # of data within get_data. Any null values we want to have removed (like on_3b, on_2b, on_1b) are done in
            # get_data using np.where
            if e != "null" and e not in labels_dict:
                # woba array will only fill with the number of possible outcomes (19)
                woba_array[counter] = float(woba_column[i])

                labels_dict[e] = counter
                labels_col_copy = np.where(labels_col_copy == e, counter, labels_col_copy)
                counter += 1
    else:
        labels_col_copy = np.copy(woba_column)
        for i, e in enumerate(woba_column):
            # we don't want to remove null values since we need to know they're there in order to delete the whole row
            # of data within get_data. Any null values we want to have removed (like on_3b, on_2b, on_1b) are done in
            # get_data using np.where
            if e != "null" and e not in labels_dict:
                # woba array will only fill with the number of possible outcomes (19)
                woba_array[counter] = float(woba_column[i])

                labels_dict[e] = counter
                labels_col_copy = np.where(labels_col_copy == e, counter, labels_col_copy)
                counter += 1

    return labels_col_copy, labels_dict, woba_array
