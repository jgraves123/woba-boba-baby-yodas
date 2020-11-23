import numpy as np
import tensorflow as tf
import csv

def get_data(data_file):
    """
    Takes in csv file and reads it to gather training and testing data and labels.
    :param data_file: file path of the csv data
    :return: Tuple of data containing:
    (
    """
    data_dict = {}
    # column titles in the CSV we want to use
    columns_we_want = ['pitch_type', 'batter', 'pitcher', 'events', 'stand', 'p_throws', 'home_team', 'away_team',
                       'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
                       'woba_value', 'bat_score', 'fld_score', 'if_fielding_alignment', 'of_fielding_alignment']
    # initialize empty lists in dictionary to add to, where keys are column titles
    for col in columns_we_want:
        data_dict[col] = []
                # read the CSV file and append each piece of data that we want from relevant columns to dictionary's relevant value
    # list
    csv_reader = csv.DictReader(open(data_file))
    for row in csv_reader:
        for col in columns_we_want:
            data_dict[col].append(row[col])
    for col in columns_we_want:
        data_dict[col] = np.array(data_dict[col])

    print("Done Creating Columns")


    # convert each list of data representing a column to an np array

    # there are 11 pitch types to encode to (plus nulls, which we filter out later)
    # changing these pitch type strings into numbers 
    # Question: will the model try to learn relationships between numbers. I.e the fact that 1 is closer 
    # to 3 than 8 is? 
    
   
    data_dict['pitch_type'] = build_ids(data_dict['pitch_type'])
    data_dict['batter'] = build_ids(data_dict['batter'])
    print(data_dict['pitcher'], " pitcher_ids before")
    data_dict['pitcher'] = build_ids(data_dict['pitcher'])
    print(data_dict['pitcher'], ' pitcher_ids after')
    data_dict['events'] = build_ids((data_dict['events']))
    # batter stance encoded as L = 0, R = 1
    data_dict['stand'] = np.where(data_dict['stand'] == 'L', 0, 1)
    # pitcher throws encoded as L = 0, R = 1
    data_dict['p_throws'] = np.where(data_dict['p_throws'] == 'L', 0, 1)
    # TODO: what are we using this home and away team for again? @john

    data_dict['away_team'], data_dict['home_team'] = field_team(data_dict['inning_topbot'], data_dict['home_team'], data_dict['away_team'])
    # away_team = fielding team
    data_dict['away_team'] = build_ids(data_dict['away_team'])
    data_dict['home_team'] = build_ids(data_dict['home_team'])
    data_dict['inning_topbot'] = build_ids(data_dict['inning_topbot'])

    # on base has player IDs for who is on base, but null for if nobody is on base, so encode null to -1
    # is -1 one a good choice or is 0 a good choice? 
    # we might want to have indicator of if someone is one base instead of who is on base
    data_dict['on_3b'] = np.where(data_dict['on_3b'] == 'null', 0, 1)
    data_dict['on_2b'] = np.where(data_dict['on_2b'] == 'null', 0, 1)
    data_dict['on_1b'] = np.where(data_dict['on_1b'] == 'null', 0, 1)
    # bat_score = difference between scores
    data_dict['bat_score'] = data_dict['bat_score'].astype(np.int32) - data_dict['fld_score'].astype(np.int32)
    # encoded as "Standard" = 1, "Strategic" = 2, "Infield shift" = 3
    data_dict['if_fielding_alignment'] = build_ids(data_dict['if_fielding_alignment'])
    # encoded as "Standard" = 1, "Strategic" = 2, "4th outfielder" = 3
    data_dict['of_fielding_alignment'] = build_ids(data_dict['of_fielding_alignment'])
    # wobas is the labels

    print("Stacking columns ...")

    # stack all the data to one massive array
    data_whole = np.column_stack((data_dict['pitch_type'], data_dict['batter'], data_dict['pitcher'], data_dict['events'],
                                  data_dict['stand'], data_dict['p_throws'], data_dict['home_team'], data_dict['away_team'],
                                  data_dict['balls'], data_dict['strikes'], data_dict['on_3b'], data_dict['on_2b'],
                                  data_dict['on_1b'], data_dict['outs_when_up'], data_dict['inning'], data_dict['inning_topbot'],
                                  data_dict['woba_value'], data_dict['bat_score'], data_dict['fld_score'],
                                  data_dict['if_fielding_alignment'], data_dict['of_fielding_alignment']))

    # get rows that have null values that can't be encoded/ would lead to model confusion
    # pitch_types (column 0), infield shift (column 18), and outfield_shift (column 19) all have entries will null
    # data that we cannot use/encode to be meaningful data

    print("Deleting Rows...")
    rows_to_delete = []
    for row_num in range(data_whole.shape[0]):
        if data_whole[row_num][0] == 'null' or data_whole[row_num][19] == 'null' or data_whole[row_num][20] == 'null':
            rows_to_delete.append(row_num)

    # delete rows that have null values we want to remove
    data_minus_nulls = np.delete(data_whole, rows_to_delete, axis=0)

    # separate out labels from data and remove the labels from the last column of the data array
    labels = (data_minus_nulls[:, 16].astype(np.float32), data_minus_nulls[:, 3].astype(np.float32))
    data_minus_nulls = np.delete(data_minus_nulls, [16, 3, 15], axis=1).astype(np.int32)

    print("Done Preprocessing!")
    return data_minus_nulls, labels 

def build_ids(column_data):
    # tokens = []
    # for s in sentences:
    #     tokens.extend(s)
    # all_words = sorted(list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

    # vocab = {word: i for i, word in enumerate(all_words)}

    # return vocab, vocab[PAD_TOKEN]
    'param: column_data is the data from one column in the data_dict dictionary in get_data'
    unique_values_list = np.unique(column_data)
    counter = 0
    for option in unique_values_list:
        if option != "null":
            column_data = np.where(column_data == option, counter, column_data)
            counter += 1
    # print(column_data, " this is column data in the function")
    return column_data

def field_team(top_bot, home, away):
    field = np.asarray(top_bot)
    hit = np.asarray(top_bot)
    field = np.where(top_bot == 'Top', home, away)
    hit = np.where(top_bot == 'Bot', away, home)
    return field, hit



get_data('full_2020_data.csv')