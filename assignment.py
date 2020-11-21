import os
import numpy as np
import tensorflow as tf
from preprocess import *
import sys


class Model(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()


    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs:
        :return: probs:
        """
        pass

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
    pass


if __name__ == '__main__':
    main()