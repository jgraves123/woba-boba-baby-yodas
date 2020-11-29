import os
import numpy as np
import tensorflow as tf
from preprocess import *
import sys


class Model(tf.keras.Model):
    def __init__(self, inputs, index_dict, max_dict):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()

        self.index_dict = index_dict
        self.max_dict = max_dict

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

        # embedding lookups
        self.E_batter = tf.Variable(
            tf.random.truncated_normal([self.num_batters, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_pitcher = tf.Variable(
            tf.random.truncated_normal([self.num_pitchers, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_team = tf.Variable(
            tf.random.truncated_normal([self.num_teams, self.embedding_size], stddev=.1, dtype=tf.float32))
        # lifting layers
        # TODO: Ask ritchie if we need this/is there a better way to do lifting? Compressing to 1d vectors?
        self.lifting_output_size = inputs.shape[0]
        self.p_lifting = tf.keras.layers.Dense(self.lifting_output_size)
        self.b_lifting = tf.keras.layers.Dense(self.lifting_output_size)
        self.t_lifting = tf.keras.layers.Dense(self.lifting_output_size)

        # dense layers
        self.hidden_size = 100
        self.denseLayer0 = tf.keras.layers.Dense(self.hidden_size, activation="relu")
        # TODO: is the output just 3 for the number of possible defensive shift/no shift options? Do we include the
        #  infield shifts and outfiled shifts?
        self.denseLayer1 = tf.keras.layers.Dense(3, activation="softmax")


    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs: a batch of input data (akin to data_minus_nulls in preprocess.py)
        :return: probs: probability value for a given configuration of the batch
        """
        # perform embedding lookups and lift to same space as the rest of the data
        b_embedding = tf.nn.embedding_lookup(self.E_batter, inputs[:, self.index_dict['batter']])
        b_lifted = self.b_lifting(b_embedding)
        p_embedding = tf.nn.embedding_lookup(self.E_pitcher, inputs[:, self.index_dict['pitcher']])
        p_lifted = self.p_lifting(p_embedding)
        t_embedding = tf.nn.embedding_lookup(self.E_team, inputs[:, self.index_dict['away_team']])
        t_lifted = self.t_lifting(t_embedding)

        # create one hot vectors of relevant categories
        pitch_type = tf.one_hot(inputs[:, self.index_dict['pitch_type']], self.num_pitch_types)
        if_alignment = tf.one_hot(inputs[:, self.index_dict['if_fielding_alignment']], self.num_if_alignments)
        of_alignment = tf.one_hot(inputs[:, self.index_dict['of_fielding_alignment']], self.num_of_alignments)

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

        # concatenate all inputs into an array of the same size
        new_input = tf.concat([b_lifted, p_lifted, t_lifted, pitch_type, if_alignment, of_alignment, balls, strikes,
                               outs, batter_stand, p_throws, on_3b, on_2b, on_1b, inning, score_dif], axis=-1)

        # run dense layers
        out0 = self.denseLayer0(new_input)
        probs = self.denseLayer1(out0)
        return probs

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
    data, labels, index_dict, max_dict = get_data("full_2020_data.csv")
    model = Model(data, index_dict, max_dict)
    print("Model Initialized...")
    model.call(data)
    print("Model.call done")

    # TODO: make test, train -> maybe in preprocess?


if __name__ == '__main__':
    main()