# woba-boba-baby-yodas
2020 DL Project

There is growing interest in applying deep learning models towards improving athlete and team performance (particularly in stat heavy sports like baseball). We built a model that calculates the value of a game state for the hitting team using wOBA (weighted on-base average). Our results could be used by the fielding team to select how to position their fielder.

Preprocess.py:

This File preprocesses our Statcast CSV file that has 89 values for all 60,000+ at-bats over the course of the 2020 season. 
Using data that could be obtained pre-pitch our model we converted categorical data (such as pitcher ID or pitch type) into integer IDs. 
We then created Labels from the outcome of the play (home run, strikeout, etc.) and the wOBA value of that event and removed any data entries with null values.
Finally we split our data into 90% training and 10% testing sets. 

assignment.py:

To create our model input we created a 1D array with embedding matrices for batter, pitcher and fielding team, one-hotted vectors for categorical variables and values of the quantitative data.
Our Three feed-forward dense layers consisted of keras layers with ReLu and softmax activations. They outputed a probability distribution over all play outcomes with distinct wOBA values. 
A sigmoidal focal cross entropy loss function was used to mitigate overfitting and optimize for a probability distribution, not the most likely outcome.
We trained for 3 epochs. 

Model Results:

Our Model:

Loss: 0.149

R-Squared: 0.037

P-value: 10^-40


