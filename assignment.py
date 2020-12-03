import tensorflow as tf
from preprocess import *
from scipy import stats


"""
Training our model:
The model seems to be getting general trends correct but we are running into two problems that seem to be contradictory:
1) The results have a ton of varience. Each time we run it the predicted outcomes for a given state vary wildly
2) The model seems to do worse when you increase the number of epochs -> probably due to overfitting?
Questions:
Are there certain things we could do that consider both of these difficulties? A normal response to high varience would 
seem to be to increase the number of epochs but that doesn't seem to work here.
"""


class Model(tf.keras.Model):
    def __init__(self, index_dict, max_dict, labels_dict, woba_array, all_cat=True):
        """
        The Model class predicts the wOBA value of a given hitting/pitching/fielding setup combination.
        """
        super(Model, self).__init__()

        # these are our input dictionaries with our data
        self.index_dict = index_dict
        self.max_dict = max_dict
        self.labels_dict = labels_dict

        # since some of the outputs are the same woba value (ex. strikeout vs fly out both have a woba of 0), not
        # all_cat consolidates some categories
        if all_cat:
            self.woba_value = woba_array
            self.num_possible_events = 19
        else:
            self.woba_value = [0, .7, .9, 1.25, 1.6, 2]
            self.num_possible_events = 6

        # these are the numbers of different batters, pitchers, teams, etc. in our overall dataset
        self.num_batters = max_dict['batter']
        self.num_pitchers = max_dict['player_name']
        self.num_teams = max_dict['away_team']
        self.num_pitch_types = max_dict['pitch_type']
        self.num_if_alignments = max_dict['if_fielding_alignment']
        self.num_of_alignments = max_dict['of_fielding_alignment']

        # arbitrary hyperparameters - can be adjusted to improve model performance as needed 
        self.embedding_size = 50
        self.batch_size = 1000
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.dense_hidden_layer_size = 100

        # embedding lookups for batter, pitcher, and team IDs  
        self.E_batter = tf.Variable(
            tf.random.truncated_normal([self.num_batters, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_pitcher = tf.Variable(
            tf.random.truncated_normal([self.num_pitchers, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.E_team = tf.Variable(
            tf.random.truncated_normal([self.num_teams, self.embedding_size], stddev=.1, dtype=tf.float32))

        # initializing two dense layers
        self.denseLayer0 = tf.keras.layers.Dense(self.dense_hidden_layer_size, activation="relu")
        self.denseLayer1 = tf.keras.layers.Dense(self.dense_hidden_layer_size, activation="relu")
        self.denseLayer2 = tf.keras.layers.Dense(self.num_possible_events, activation="softmax")

    def call(self, inputs):
        """
        Runs model layers and embedding calculations that will be used to update gradients in train() and test()
        :param inputs: a batch of input data (akin to data_minus_nulls in preprocess.py)
        :return: probs: probability value for a given configuration of the batch
        """
        # perform embedding lookups on our batter, pitcher, and team IDs 
        b_embedding = tf.nn.embedding_lookup(self.E_batter, inputs[:, self.index_dict['batter']])
        p_embedding = tf.nn.embedding_lookup(self.E_pitcher, inputs[:, self.index_dict['player_name']])
        t_embedding = tf.nn.embedding_lookup(self.E_team, inputs[:, self.index_dict['away_team']])

        # create one hot vectors of pitch type, infield alignment, and outfield alignment since they have numerical
        # values but currently those values should not be assigned any meaning
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
        out1 = self.denseLayer1(out0)
        probs = self.denseLayer2(out1)

        # return probabilities
        return probs

    def loss_mean_square(self, probs, labels):
        """
        Calculates average loss of the prediction
        :param probs: probabilities from our model.call function for play outcomes
        :param labels: correct play outcomes
        :return: the mean squared error loss of the model as a tensor of size 1
        """
        probs_value = self.woba_calc(probs)
        real_woba = tf.convert_to_tensor(labels[:, 0])
        loss = tf.reduce_mean(tf.square(probs_value - real_woba))*100
        return loss

    # TODO: delete when we confirm we don't need this anymore
    def loss_event(self, probs, labels):
        """
        Calculates average loss of the prediction
        :param probs: proabilities from our model.call function for play outcomes
        :param labels: correct play outcomes
        :return: the cross entropy loss of the model as a tensor of size 1
        """
        new_labels = labels[:, 1]
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(new_labels, probs))

    def accuracy(self, probs, labels):
        """
        Calculates average loss of the prediction
        :param probs: probabilities of outcomes at each index, batch size x 19
        :param labels: correct labels for the event outcome of the play
        :return: the woba value our model predicts, the actual woba value
        """
        probs_value = self.woba_calc(probs)
        real_woba = labels[:, 0]
        return probs_value.numpy(), real_woba

    def woba_calc(self, probs):
        """
        Calculates woba given a probability of each outcome
        :param probs: probabilities of outcomes at each index, batch size x 19
        :return: 1D array of woba values
        """
        return tf.reduce_sum(probs * self.woba_value, axis=1)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples in batches
    :param model: the initialized model to use for forward and backward pass
    :param train_inputs: shuffled train inputs (all inputs for training) 
    :param train_labels: shuffled train labels (all labels for training) 
    :return: average loss across an epoch
    """
    # go through data batch by batch
    loss_list = []
    for x in range(0, train_inputs.shape[0], model.batch_size):
        # get inputs and labels for that batch
        batch_inputs = train_inputs[x:x+model.batch_size, :]
        batch_labels = train_labels[x:x+model.batch_size, :]

        # update gradients
        with tf.GradientTape() as tape: 
            probs = model.call(batch_inputs)
            l = model.loss_mean_square(probs, batch_labels)
            loss_list.append(l)
        gradients = tape.gradient(l, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    average_loss = tf.reduce_mean(tf.convert_to_tensor(loss_list))
    return average_loss


def test(model, test_inputs, test_labels):
    """
    Runs through all test data in batches 
    :param model: the trained model to use for prediction
    :param test_inputs: test inputs (all inputs for testing)
    :param test_labels: test labels (all labels for testing) 
    :returns: average loss across batches of testing data 
    """
    loss_list = []
    all_pred = np.array([])
    all_true = np.array([])
    # go through data batch by batch
    for x in range(0, test_inputs.shape[0], model.batch_size):
        # get inputs and labels for that batch
        batch_inputs = test_inputs[x:x+model.batch_size, :]
        batch_labels = test_labels[x:x+model.batch_size, :]

        # get loss and add to list
        probs = model.call(batch_inputs)
        l = model.loss_mean_square(probs, batch_labels)
        loss_list.append(l)
        print(l)

        # get predictions to calculate regression statistics
        pred, true = model.accuracy(probs, batch_labels)
        all_pred = np.concatenate([all_pred, np.array(pred)])
        all_true = np.concatenate([all_true, np.array(true)])

    slope, intercept, r_value, p_value, std_err = stats.linregress(all_true, all_pred)
    print("r-value: ", r_value)
    print("r-squared: ", r_value ** 2)
    print("p-value: ", p_value)

    average_loss = tf.reduce_mean(tf.convert_to_tensor(loss_list))
    return average_loss


def test_value(model, input):
    """
    Gets model's predicted woba and play outcome, used for testing in main()
    :param model: the trained model to use for prediction
    :param input: play data for a single event
    :returns: probabilities of play outcomes from the model, predicted woba score of the model
    """
    probs = model.call(input)
    pred, fake = model.accuracy(probs, np.array([[0, 0]]))
    return probs, pred


def create_test(pitcher_dict, batter_dict, team_dict, pitcher, batter, team, strikes, balls, shift):
    """
    Creates an input vector representing a play in the format model.call needs, used for testing in main()
    :param shift: whether or not defense is shifting for this at bat
    :param balls: number of balls on the play
    :param strikes: number of strikes on the play
    :param team: string abbreviation (ex. "CLE") of the fielding team
    :param batter: string name of batter at bat
    :param pitcher: string name of pitcher pitching
    :param team_dict: dictionary of team IDs to modified preprocessed team IDs
    :param batter_dict: dictionary of batter IDs to modified preprocessed batter IDs
    :param pitcher_dict: dictionary of pitcher IDs to modified preprocessed pitcher IDs
    :returns: np array of the input in the requisite format for our model
    """
    input = [0]*16
    input[0] = 0  # FF or standard fastball
    input[1] = batter_dict[batter]
    input[2] = pitcher_dict[pitcher]
    input[3] = 1  # R - batter
    input[4] = 1  # R - pitcher
    input[5] = team_dict[team]
    input[6] = balls
    input[7] = strikes
    # 8 - 11 mean 0-0 count and no one on and no one out
    input[12] = 1  # first inning
    # 13 means tied game, 14 means no inf shift
    input[15] = shift  # of shift
    return np.array([input])

# where the magic happens
def main():
    train_data, test_data, train_labels, test_labels, labels_dictionary, woba_array, index_dict, max_dict, \
        pitcher_dict, batter_dict, team_dict = get_data("full_2020_data.csv")
    model = Model(index_dict, max_dict, labels_dictionary, woba_array, True)
    print("Model initialized...")

    for j in range(2):
        val = train(model, train_data, train_labels)
        print('Epoch:', j, 'average loss:', val)
        sorting = tf.random.shuffle(range(np.size(train_labels, 0)))
        train_data = tf.gather(train_data, sorting)
        train_labels = tf.gather(train_labels, sorting)

    print("Model testing finished...")
    acc = test(model, test_data, test_labels)
    print("Model accuracy: ", acc)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Shane Bieber', '543807', 'CLE', 0, 0, 0)
    print("Springer v. Bieber")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Shane Bieber', '543807', 'CLE', 2, 0, 0)
    print("Springer v. Bieber w/ 2 strikes")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Shane Bieber', '543807', 'CLE', 0, 3, 0)
    print("Springer v. Bieber w/ 3 balls")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Shane Bieber', '666176', 'CLE', 0, 0, 0)
    print("Adell v. Bieber")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '543807', 'NYM', 0, 0, 0)
    print("Springer v. Porcello")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '666176', 'NYM', 0, 0, 0)
    print("Adell v. Porcello")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '518934', 'NYM', 0, 0, 0)
    print("Leimahiu v. Porcello no shift")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '518934', 'NYM', 0, 0, 1)
    print("Leimahiu v. Porcello shift")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '448801', 'NYM', 0, 0, 0)
    print("Davis v. Porcello no shift")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)

    value = create_test(pitcher_dict, batter_dict, team_dict, 'Rick Porcello', '448801', 'NYM', 0, 0, 1)
    print("Davis v. Porcello shift")
    probs, pred = test_value(model, value)
    print("predicted wOBA: ", pred)


if __name__ == '__main__':
    main()
