import tensorflow as tf
from preprocess import *


class Model(tf.keras.Model):
    def __init__(self, index_dict, max_dict, labels_dict):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()

        # these are our input dictionaries with our data
        self.index_dict = index_dict
        self.max_dict = max_dict
        self.labels_dict = labels_dict

        # these are the numbers of different batters, pitchers, teams, etc. in our overall dataset
        self.num_batters = max_dict['batter']
        self.num_pitchers = max_dict['pitcher']
        self.num_teams = max_dict['away_team']
        self.num_pitch_types = max_dict['pitch_type']
        self.num_if_alignments = max_dict['if_fielding_alignment']
        self.num_of_alignments = max_dict['of_fielding_alignment']

        # arbitrary hyperparameters - can be adjusted to improve model performance as needed 
        self.embedding_size = 50
        self.batch_size = 1000
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.dense_hidden_layer_size = 100

        # this is the number of different possible events (outcomes) per at bat situation (ex: out, home run, single,
        # double, etc.). There are 19 different play outcomes in our dataset
        self.num_possible_events = 19

        # embedding lookups for batter, pitcher, and team IDs  
        self.E_batter = tf.Variable(
            tf.random.truncated_normal([self.num_batters, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_pitcher = tf.Variable(
            tf.random.truncated_normal([self.num_pitchers, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_team = tf.Variable(
            tf.random.truncated_normal([self.num_teams, self.embedding_size], stddev=.1, dtype=tf.float32))

        # initializing two dense layers
        self.denseLayer0 = tf.keras.layers.Dense(self.dense_hidden_layer_size, activation="relu")
        self.denseLayer1 = tf.keras.layers.Dense(self.num_possible_events, activation="softmax")

    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs: a batch of input data (akin to data_minus_nulls in preprocess.py)
        :return: probs: probability value for a given configuration of the batch
        """

        # perform embedding lookups on our batter, pitcher, and team IDs 
        b_embedding = tf.nn.embedding_lookup(self.E_batter, inputs[:, self.index_dict['batter']])
        p_embedding = tf.nn.embedding_lookup(self.E_pitcher, inputs[:, self.index_dict['pitcher']])
        t_embedding = tf.nn.embedding_lookup(self.E_team, inputs[:, self.index_dict['away_team']])

        # create one hot vectors of pitch type, infield alignment, and outfield alignment 
        # TODO: why are we one-hotting these specific parameters? If so, why aren't we also one-hotting parameters such as
        # balls, strikes, outs, etc.?
        pitch_type = tf.one_hot(inputs[:, self.index_dict['pitch_type']], self.num_pitch_types)
        if_alignment = tf.one_hot(inputs[:, self.index_dict['if_fielding_alignment']], self.num_if_alignments)
        of_alignment = tf.one_hot(inputs[:, self.index_dict['of_fielding_alignment']], self.num_of_alignments)

        # previous lines were working with categories that were in the form of ints to perform lookups and do
        # one-hotting. However, now we must cast the inputs to floats in order to perform concatenation
        inputs = tf.cast(inputs, dtype=tf.float32)

        # grabbing data from corresponding columns of parameters and expanding dims to make them compatible for
        # concatenation
        balls = inputs[:, self.index_dict['balls']]
        balls = tf.expand_dims(balls, axis=-1)

        strikes = inputs[:, self.index_dict['strikes']]
        strikes = tf.expand_dims(strikes, axis=-1)

        outs = inputs[:, self.index_dict['outs_when_up']]
        outs = tf.expand_dims(outs, axis=-1)

        batter_stand = inputs[:, self.index_dict['stand']]
        batter_stand = tf.expand_dims(batter_stand, axis=-1)

        p_throws = inputs[:, self.index_dict['p_throws']]
        p_throws = tf.expand_dims(p_throws, axis=-1)

        on_3b = inputs[:, self.index_dict['on_3b']]
        on_3b = tf.expand_dims(on_3b, axis=-1)

        on_2b = inputs[:, self.index_dict['on_2b']]
        on_2b = tf.expand_dims(on_2b, axis=-1)

        on_1b = inputs[:, self.index_dict['on_1b']]
        on_1b = tf.expand_dims(on_1b, axis=-1)

        inning = inputs[:, self.index_dict['inning']]
        inning = tf.expand_dims(inning, axis=-1)

        score_dif = inputs[:, self.index_dict['bat_score']]
        score_dif = tf.expand_dims(score_dif, axis=-1)

        # concatenate all of the above data into a massive array of batch_size x 171 
        new_input = tf.concat([b_embedding, p_embedding, t_embedding, pitch_type, if_alignment, of_alignment, balls,
                               strikes, outs, batter_stand, p_throws, on_3b, on_2b, on_1b, inning, score_dif], axis=-1)

        # pass new_input through our two dense layers
        out0 = self.denseLayer0(new_input)
        probs = self.denseLayer1(out0)

        # return probabilities
        return probs

    # TODO: fill out.
    def loss(self, probs, labels):
        """
        Calculates average loss of the prediction
        :param probs:
        :param labels:
        :return: the loss of the model as a tensor of size 1
        """
        pass


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples in batches
    :param model: the initialized model to use for forward and backward pass
    :param train_inputs: shuffled train inputs (all inputs for training) 
    :param train_labels: shuffled train labels (all labels for training) 
    :return: None
    """
    # go through data batch by batch
    for x in range(0, train_inputs.shape[0], model.batch_size):
        # get inputs and labels for that batch
        batch_inputs = train_inputs[x:x+model.batch_size, :]
        batch_labels = train_labels[x:x+model.batch_size, :]

        # update gradients
        with tf.GradientTape() as tape: 
            probs = model.call(batch_inputs)
            l = model.loss(probs, batch_labels)
        gradients = tape.gradient(l, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through all test data in batches 
    :param model: the trained model to use for prediction
    :param test_inputs: test inputs (all inputs for testing)
    :param test_labels: test labels (all labels for testing) 
    :returns: average loss across batches of testing data 
    """

    loss_list = []
    # go through data batch by batch
    for x in range(0, test_inputs.shape[0], model.batch_size):
        # get inputs and labels for that batch
        batch_inputs = test_inputs[x:x+model.batch_size, :]
        batch_labels = test_labels[x:x+model.batch_size, :]

        # get loss and add to list
        probs = model.call(batch_inputs)
        l = model.loss(probs, batch_labels)
        loss_list.append(l)

    average_loss = tf.reduce_mean(tf.convert_to_tensor(loss_list), dtype=tf.float32)
    return average_loss


# where the magic happens
def main():
    train_data, test_data, train_labels, test_labels,  labels_dictionary, index_dict, max_dict = get_data("full_2020_data.csv")
    model = Model(index_dict, max_dict, labels_dictionary)
    print("Model initialized...")

    train(model, train_data, train_labels)
    print("Model training finished...")

    acc = test(model, test_data, test_labels)
    print("Model testing finished...")
    print("Model accuracy: ", acc)


if __name__ == '__main__':
    main()
