import os
import numpy as np
import tensorflow as tf
from preprocess import *
import sys


class Model(tf.keras.Model):
    def __init__(self, index_dict, max_dict, labels_dict):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()

        self.index_dict = index_dict
        self.max_dict = max_dict
        self.labels_dict = labels_dict

        # highest id # = size of arrays
        self.num_batters = max_dict['batter']
        self.num_pitchers = max_dict['pitcher']
        self.num_teams = max_dict['away_team']
        self.num_pitch_types = max_dict['pitch_type']
        self.num_if_alignments = max_dict['if_fielding_alignment']
        self.num_of_alignments = max_dict['of_fielding_alignment']
        

        # arbitrary hyperparameters
        self.embedding_size = 50
        self.batch_size = 1000
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.num_possible_events = 19

        # embedding lookups
        self.E_batter = tf.Variable(
            tf.random.truncated_normal([self.num_batters, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_pitcher = tf.Variable(
            tf.random.truncated_normal([self.num_pitchers, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_team = tf.Variable(
            tf.random.truncated_normal([self.num_teams, self.embedding_size], stddev=.1, dtype=tf.float32))
        
        # dense layers
        self.hidden_size = 100
        self.denseLayer0 = tf.keras.layers.Dense(self.hidden_size, activation="relu")
        # TODO: is the output just 3 for the number of possible defensive shift/no shift options? Do we include the
        #  infield shifts and outfiled shifts?
        self.denseLayer1 = tf.keras.layers.Dense(self.num_possible_events, activation="softmax")


    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs: a batch of input data (akin to data_minus_nulls in preprocess.py)
        :return: probs: probability value for a given configuration of the batch
        """

        # perform embedding lookups and lift to same space as the rest of the data
        b_embedding = tf.nn.embedding_lookup(self.E_batter, inputs[:, self.index_dict['batter']])
        p_embedding = tf.nn.embedding_lookup(self.E_pitcher, inputs[:, self.index_dict['pitcher']])
        t_embedding = tf.nn.embedding_lookup(self.E_team, inputs[:, self.index_dict['away_team']])
        

        # create one hot vectors of relevant categories
        pitch_type = tf.one_hot(inputs[:, self.index_dict['pitch_type']], self.num_pitch_types)
        if_alignment = tf.one_hot(inputs[:, self.index_dict['if_fielding_alignment']], self.num_if_alignments)
        of_alignment = tf.one_hot(inputs[:, self.index_dict['of_fielding_alignment']], self.num_of_alignments)

        inputs = tf.cast(inputs, dtype=tf.float32)
        # save as variables for tf.concat below
        balls = inputs[:, self.index_dict['balls']]
        strikes = inputs[:, self.index_dict['strikes']]
        outs = inputs[:, self.index_dict['outs_when_up']]
        batter_stand = inputs[:, self.index_dict['stand']]
        p_throws = inputs[:, self.index_dict['p_throws']]
        on_3b = inputs[:, self.index_dict['on_3b']]
        on_2b = inputs[:, self.index_dict['on_2b']]
        on_1b = inputs[:, self.index_dict['on_1b']]
        inning = inputs[:, self.index_dict['inning']]
        score_dif = inputs[:, self.index_dict['bat_score']]
        balls = tf.expand_dims(balls, axis=-1)
        strikes = tf.expand_dims(strikes, axis=-1)
        outs = tf.expand_dims(outs, axis=-1)
        batter_stand = tf.expand_dims(batter_stand, axis=-1)
        p_throws = tf.expand_dims(p_throws, axis=-1)
        on_3b = tf.expand_dims(on_3b, axis=-1)
        on_2b = tf.expand_dims(on_2b, axis=-1)
        on_1b = tf.expand_dims(on_1b, axis=-1)
        inning = tf.expand_dims(inning, axis=-1)
        score_dif = tf.expand_dims(score_dif, axis=-1)
        
        print(balls.shape, ' balls shape post expanding balls')

        # concatenate all inputs into an array of the same size
        new_input = tf.concat([b_embedding, p_embedding, t_embedding, pitch_type, if_alignment, of_alignment, balls, strikes,
                               outs, batter_stand, p_throws, on_3b, on_2b, on_1b, inning, score_dif], axis=-1)
        print(new_input.shape, " new_input shape")
        # new_input = tf.reshape(new_input, [self.batch_size, -1])
        # print(new_input.shape, " new_input shape post reshape")
        # run dense layers
        out0 = self.denseLayer0(new_input)
        probs = self.denseLayer1(out0)
        return probs

# [1,2,3]  [2,3,4] -->  [1,2,3,2,3,4]

    def loss(self, probs, labels):
        """
        Calculates average loss of the prediction
        :param probs:
        :param labels:
        :return: the loss of the model as a tensor of size 1
        """
        pass


def train(model, training_data):
    pass


def test(model, testing_data):
    pass


def main():
    train_data, test_data, train_labels, test_labels, index_dict, max_dict, labels_dictionary = get_data("full_2020_data.csv")
    model = Model(index_dict, max_dict, labels_dictionary)
    print("Model Initialized...")
    model.call(train_data)
    print("Model.call finished")

    # TODO: make test, train -> maybe in preprocess?


if __name__ == '__main__':
    main()