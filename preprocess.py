import numpy as np
import tensorflow as tf
import csv

def get_data(data_file):
    """
    Takes in csv file and reads it to gather training and testing data and labels.
    :param data_file: file path of the csv data
    :return: data from relevant columns of csv as np array, tuple of (woba values, events mapped as int IDs)
    """
    data_dict = {}
    # column titles in the CSV we want to use
    columns_we_want = np.array(['pitch_type', 'batter', 'pitcher', 'events', 'stand', 'p_throws', 'home_team', 'away_team',
                       'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
                       'woba_value', 'bat_score', 'fld_score', 'if_fielding_alignment', 'of_fielding_alignment'])

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

    print("Done Creating Columns")

    # build IDs maps string data in certain columns to IDs using build_ids function
    data_dict['pitch_type'] = build_ids(data_dict['pitch_type'])
    data_dict['batter'] = build_ids(data_dict['batter'])
    data_dict['pitcher'] = build_ids(data_dict['pitcher'])
    data_dict['events'] = build_ids((data_dict['events']))
    data_dict['if_fielding_alignment'] = build_ids(data_dict['if_fielding_alignment'])
    data_dict['of_fielding_alignment'] = build_ids(data_dict['of_fielding_alignment'])

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
    # away_team key now represents the fielding team name and home_team key now represents the hitting team name, which
    # we now map to IDs
    data_dict['away_team'] = build_ids(data_dict['away_team'])
    # don't actually use these two columns, but leaving it in in case we want it later, and need to do it so typecasting
    # and operations later don't cause issues
    data_dict['home_team'] = build_ids(data_dict['home_team'])
    data_dict['inning_topbot'] = build_ids(data_dict['inning_topbot'])

    # bat_score column now represents the difference between scores of hitting and fielding team
    data_dict['bat_score'] = data_dict['bat_score'].astype(np.int32) - data_dict['fld_score'].astype(np.int32)

    print("Stacking columns ...")

    # stack all the data to one massive array
    data_whole = np.column_stack((data_dict['pitch_type'], data_dict['batter'], data_dict['pitcher'],
                                  data_dict['events'], data_dict['stand'], data_dict['p_throws'],
                                  data_dict['home_team'], data_dict['away_team'], data_dict['balls'],
                                  data_dict['strikes'], data_dict['on_3b'], data_dict['on_2b'], data_dict['on_1b'],
                                  data_dict['outs_when_up'], data_dict['inning'], data_dict['inning_topbot'],
                                  data_dict['woba_value'], data_dict['bat_score'], data_dict['fld_score'],
                                  data_dict['if_fielding_alignment'], data_dict['of_fielding_alignment']))

    print("Deleting Rows...")

    # get rows that have null values that can't be encoded/ would lead to model confusion
    # pitch_types (column 0), infield shift (column 18), and outfield_shift (column 19) all have entries will null
    # data that we cannot use/encode to be meaningful data
    rows_to_delete = []
    for row_num in range(data_whole.shape[0]):
        if data_whole[row_num][0] == 'null' or data_whole[row_num][19] == 'null' or data_whole[row_num][20] == 'null':
            rows_to_delete.append(row_num)

    # delete rows that have null values we want to remove
    data_minus_nulls = np.delete(data_whole, rows_to_delete, axis=0)

    # separate out labels from data and remove the labels from the data array, also remove inning_topbot since we don't
    # want to use it at the current moment
    labels = (data_minus_nulls[:, 16].astype(np.float32), data_minus_nulls[:, 3].astype(np.float32))
    data_minus_nulls = np.delete(data_minus_nulls, [16, 3, 15], axis=1).astype(np.int32)
    update_columns = np.delete(columns_we_want, [16, 3, 15], axis=0)

    # index dict is a dictionary between column name and index number of column
    index_dict = {}
    for i, e in enumerate(update_columns):
        index_dict[e] = i

    # max dict is a dictionary between column name and number of unique values
    # used in assignment.py to get number of pitchers, batters, etc.
    max_dict = {}
    for i, e in enumerate(update_columns):
        max_dict[e] = np.amax(data_minus_nulls[:, i]) + 1

    print("Done Preprocessing!")

    return data_minus_nulls, labels, index_dict, max_dict


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
