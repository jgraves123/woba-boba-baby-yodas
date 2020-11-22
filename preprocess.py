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

    # convert each list of data representing a column to an np array

    # there are 11 pitch types to encode to (plus nulls, which we filter out later)
    # changing these pitch type strings into numbers 
    # Question: will the model try to learn relationships between numbers. I.e the fact that 1 is closer 
    # to 3 than 8 is? 
    
    pitchTypesList = ['CH', 'CS', 'CU', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'SI', 'SL']
    # we chose to not work with a 0 index for this list 
    counter = 1
    pitch_types = np.array(data_dict['pitch_type'])
    #where you turn the strings into the corresponding numbers 
    for pitch in pitchTypesList:
        pitch_types = np.where(pitch_types == pitch, counter, pitch_types)
        counter += 1
    batter_ids = np.array(data_dict['batter'])
    pitcher_ids = np.array(data_dict['pitcher'])
    # TODO: encode play_outcomes 1-19 similar to how we did pitch_types
    play_outcomes = np.array(data_dict['events'])
    playEventsList =  ['double', 'double_play', 'field_error', 'field_out', 'fielders_choice', 'fielders_choice_out', 'force_out',
    'grounded_into_double_play', 'hit_by_pitch', 'home_run', 'sac_bunt', 'sac_fly', 'sac_fly_double_play', 'single', 
    'strikeout', 'strikeout_double_play', 'triple', 'triple_play', 'walk']
    counter = 1
    for event in playEventsList:
        play_outcomes = np.where(play_outcomes == event, counter, play_outcomes)
        counter += 1
    print(play_outcomes, "this is play outcomes")
    # batter stance encoded as L = 1, R = 2
    batter_stance = np.array(data_dict['stand'])
    batter_stance = np.where(batter_stance == 'L', 1, 2)
    # pitcher handedness similarly encoded as L = 1, R = 2
    pitcher_handedness = np.array(data_dict['p_throws'])
    pitcher_handedness = np.where(pitcher_handedness == 'L', 1, 2)
    # TODO: what are we using this home and away team for again? @john
    home_team = np.array(data_dict['home_team'])
    away_team = np.array(data_dict['away_team'])
    balls = np.array(data_dict['balls'])
    # number of strikes that person at bat has 
    strikes = np.array(data_dict['strikes'])
    # on base has player IDs for who is on base, but null for if nobody is on base, so encode null to -1
    # is -1 one a good choice or is 0 a good choice? 
    on_3b = np.array(data_dict['on_3b'])
    on_3b = np.where(on_3b == 'null', -1, on_3b)
    on_2b = np.array(data_dict['on_2b'])
    on_2b = np.where(on_3b == 'null', -1, on_2b)
    on_1b = np.array(data_dict['on_1b'])
    on_1b = np.where(on_3b == 'null', -1, on_1b)
    outs = np.array(data_dict['outs_when_up'])
    innings = np.array(data_dict['inning'])
    # TODO how are we using this with home and away team
    innings_top_or_bot = np.array(data_dict['inning_topbot']) # ask john
    batting_team_score = np.array(data_dict['bat_score'])
    pitching_team_score = np.array(data_dict['fld_score'])
    # encoded as "Standard" = 1, "Strategic" = 2, "Infield shift" = 3
    infield_shifts = np.array(data_dict['if_fielding_alignment'])
    infield_shifts = np.where(infield_shifts == 'Standard', 1, infield_shifts)
    infield_shifts = np.where(infield_shifts == 'Strategic', 2, infield_shifts)
    infield_shifts = np.where(infield_shifts == 'Infield shift', 3, infield_shifts)
    # encoded as "Standard" = 1, "Strategic" = 2, "4th outfielder" = 3
    outfield_shifts = np.array(data_dict['of_fielding_alignment'])
    outfield_shifts = np.where(outfield_shifts == 'Standard', 1, outfield_shifts)
    outfield_shifts = np.where(outfield_shifts == 'Strategic', 2, outfield_shifts)
    outfield_shifts = np.where(outfield_shifts == '4th outfielder', 3, outfield_shifts)
    # wobas is the labels
    wobas = np.array(data_dict['woba_value'])

    # stack all the data to one massive array
    data_whole = np.column_stack((pitch_types, batter_ids, pitcher_ids, play_outcomes, batter_stance, pitcher_handedness,
                                  home_team, away_team, balls, strikes, on_3b, on_2b, on_1b, outs, innings, innings_top_or_bot,
                                  batting_team_score, pitching_team_score, infield_shifts, outfield_shifts, wobas))

    # get rows that have null values that can't be encoded/ would lead to model confusion
    # pitch_types (column 0), infield shift (column 18), and outfield_shift (column 19) all have entries will null
    # data that we cannot use/encode to be meaningful data
    rows_to_delete = []
    for row_num in range(data_whole.shape[0]):
        if data_whole[row_num][0] == 'null' or data_whole[row_num][18] == 'null' or data_whole[row_num][19] == 'null':
            rows_to_delete.append(row_num)

    # delete rows that have null values we want to remove
    data_minus_nulls = np.delete(data_whole, rows_to_delete, axis=0)

    # separate out labels from data and remove the labels from the last column of the data array
    labels = data_minus_nulls[:, 20]
    data_minus_nulls = np.delete(data_minus_nulls, 20, axis=1)
    # TODO there may be some type issues here with blanket casting to float 32, examine and make sure it's fine
    # cast encoded data to float32
    # data_final = data_minus_nulls.astype(np.float32)
    print(data_minus_nulls, "data_minus_nulls")
    print(labels, 'labels')
    labels = labels.astype(np.float32)
    print(labels, 'labels')
    return data_minus_nulls, labels 

get_data('full_2020_data.csv')