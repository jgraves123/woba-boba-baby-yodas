import os
import numpy as np
import tensorflow as tf
from preprocess import *
import sys


class Model(tf.keras.Model):
    def __init__(self, index_dict, max_dict):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()

        self.index_dict = index_dict

        self.num_batters = max_dict['batter']
        self.num_pitchers = max_dict['pitcher']
        self.num_teams = max_dict['away_team']
        self.num_pitch_types = max_dict['pitch_type']
        self.num_if_alignments = max_dict['if_fielding_alignment']
        self.num_of_alignments = max_dict['of_fielding_alignment']

        self.embedding_size = 50
        self.batch_size = 1000

        self.E_batter = tf.Variable(
            tf.random.truncated_normal([self.num_batters, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_pitcher = tf.Variable(
            tf.random.truncated_normal([self.num_pitchers, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_team = tf.Variable(
            tf.random.truncated_normal([self.num_teams, self.embedding_size], stddev=.1, dtype=tf.float32))

        # TODO: initialize dense layers


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs:
        :return: probs:
        """
        b_embedding = tf.nn.embedding_lookup(self.E_batter, inputs[:, self.index_dict['batter']])
        p_embedding = tf.nn.embedding_lookup(self.E_pitcher, inputs[:, self.index_dict['pitcher']])
        t_embedding = tf.nn.embedding_lookup(self.E_team, inputs[:, self.index_dict['away_team']])
        pitch_type = tf.one_hot(inputs[:, self.index_dict['pitch_type']], self.num_pitch_types)
        if_alignment = tf.one_hot(inputs[:, self.index_dict['if_fielding_alignment']], self.num_if_alignments)
        of_alignment = tf.one_hot(inputs[:, self.index_dict['of_fielding_alignment']], self.num_of_alignments)
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

        new_input = tf.concat([b_embedding, p_embedding, t_embedding, pitch_type, if_alignment, of_alignment, balls,
                               strikes, outs, batter_stand, p_throws, on_3b, on_2b, on_1b, inning, score_dif], axis=-1)

        # TODO: run dense layers


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

    model = Model(index_dict, max_dict)

    # TODO: make test, train -> maybe in preprocess?


if __name__ == '__main__':
    main()