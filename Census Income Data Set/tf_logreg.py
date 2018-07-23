import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn import svm

DATA_DIR = './data/'
DATA_FILE = './adult.data'
TEST_FILE = './adult.test'
NAMES = ['age', 'workclass','fnlwgt','education','education-num','marital-status',
         'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']


def preprocess_data(data_df, feature_list=None, mean=None, std=None):
    d = {}
    
    # Number of samples
    d['num_samples'] = len(data_df)
    
    # Replace '?' with NaN
    data_df = data_df.replace(' ?', np.nan)
    
    # Find how many rows have missing values, indicated by a '?'
    element_idx_missing_data = data_df.isnull()
    row_idx_missing_data = element_idx_missing_data.any(axis=1)
    num_samples_missing_data = sum(row_idx_missing_data)
    d['num_samples_missing_data'] = num_samples_missing_data
    
    # Remove samples with missing data
    data_df = data_df.dropna()
    
    # We can remove the eduction column as this is already nicely encoded
    # numerically in the education-num column.
    data_df = data_df.drop('education', axis=1)
    
    # Now we need to deal with the other categorical features which aren't
    # easily encoded as numerical values. We choose to use one-hot encoding,
    # because it will avoid any "confusion" that the algorithm may have
    # with numerical encodings of categories. The one-hot encoding will
    # effectively "unroll" all of the categorical features into unique features
    # corresponding to their possible values. For example the 'sex' feature
    # can take values 'Male' and 'Female'. This feature will become a 2D vector
    # ['sex_Male', 'sex_Female']; a feature [0, 1] denotes a 'Female' sample.
    # We want to get one-hot vectors for: sex, workclass, marital-status, occupation,
    # relationship, race, native-country, income.
    # Use a funky separator ('$_$') to be able to pick out the one-hot values later.
    dummy_cols = ['sex','workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'income']
    data_df = pd.get_dummies(data_df,columns=dummy_cols, prefix_sep='$_$')
    
    # Extract the target vector: income. We previously transformed this into a one-hot,
    # however we want this to be a binary scalar. The one-hot transformation will
    # help us to easily make a binary value using boolean operators.
    # Income <= 50K -> 0, income > 50K -> 1.
    try:
        data_df.rename(columns = {'income$_$ >50K.':'income$_$ >50K'}, inplace = True)
        data_df.rename(columns = {'income$_$ <=50K.':'income$_$ <=50K'}, inplace = True)
    except:
        pass
    target = data_df['income$_$ >50K']
    
    # Now drop the income vectors from the data frame to leave us with just the 
    # features.
    data_df = data_df.drop(['income$_$ <=50K', 'income$_$ >50K'], axis=1)
    
    d['feature_list'] = list(data_df)
    
    if feature_list is not None:
        # If there features are missing, add them
        for x in feature_list:
            if x not in list(data_df):
                data_df[x] = pd.Series([0 for _ in range(len(data_df))])
        for x in list(data_df):
            if x not in feature_list:
                data_df.drop(x)
        # Rearrange to ensure that the features are in the same order
        data_df = data_df[feature_list]

    # Since the data is a collection of one-hots and continuous values of different
    # ranges, let's do some normalisation and scaling.
    # We'll try two different normalisation schemes: normalise everything, and
    # only normalise continuous (non-one-hot) values.=
    data_mean = data_df.mean() if (mean is None) else mean
    data_std = data_df.std() if (std is None) else std
    data_df_norm_all = data_df - data_mean
    data_df_norm_all = data_df_norm_all / data_std
    d['mean'] = data_mean
    d['std'] = data_std
    # Do the continuous columns in a dumb way
    #non_oh_cols = [x for x in list(data_df) if '$_$' not in x]
    #data_df_norm_cont = data_df.copy()
    #for col in  non_oh_cols:
    #    data_df_norm_cont[col] = data_df[col] - data_df[col].mean()
    #    data_df_norm_cont[col] = data_df_norm_cont[col] / data_df[col].std()
        
    # We now have two normalised feature sets. Let's do some training.
    # First extend the features to have a column of 1s (bias) in the first column
    data_df_norm_all.insert(0, 'bias', [1 for _ in range(len(data_df_norm_all))])
    #data_df_norm_cont.insert(0, 'bias', [1 for _ in range(len(data_df_norm_cont))])
    
    # Make features matrix
    features = data_df_norm_all.as_matrix()
    target = np.array(target, ndmin=2).T
    d['features'] = features
    d['target'] = target
    
    return d


# TODO: Implement logistic regression with TF
class TFLogisticRegression():
    def __init__(self, learning_rate, num_features, num_classes):
        
        # Classifier graph
        self.input_node = tf.placeholder(dtype=tf.float32,
                          shape=[num_features, None],
                          name="feature_ph")
        self.theta = tf.Variable(dtype=tf.float32,
                       initial_value=tf.zeros([num_features, 1]),
                       trainable=True,
                       name="theta")
        self.prediction = tf.sigmoid(tf.matmul(self.input_node,
                                               self.theta,
                                               transpose_a=True),
                                     "prediction_op")
        
        # Learning graph
        self.target = tf.placeholder(dtype=tf.float32,
                                     shape=[None,num_classes],
                                     name="target_ph")
        #a = tf.multiply(self.target, tf.log(self.prediction))
        #b = tf.multiply(1-self.target, tf.log(1.0-self.prediction))
        #self.cost = tf.reduce_mean(-tf.reduce_sum(a + b, reduction_indices=1),
        #                           name="cost_op")
        self.cost = tf.losses.sigmoid_cross_entropy(self.target,
                                                    logits=self.prediction)
        #self.update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.update = self.update.minimize(self.cost)

    def initialise_graph(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        
    def predict(self, sess, x):
        return sess.run(self.prediction,
                        feed_dict={self.input_node:x})
    
    
    def _train_step(self, sess, _x, _y):
        feed_dict = {self.input_node: _x,
                     self.target: _y}
        cost, _ = sess.run([self.cost, self.update],
                           feed_dict=feed_dict)
        return cost
    
    def train(self, sess, x, y, batch, epochs):
        num_batches = int(np.floor(x.shape[0]/batch))
        batch_remainder = x.shape[0] % batch
        cost_list = []
        for i in tqdm(range(epochs)):
            for j in range(num_batches):
                k = j * batch
                _x = np.matrix(x[k:k+batch-1]).T
                _y = np.matrix(y[k:k+batch-1])
                cost = self._train_step(sess, _x, _y)
                cost_list.append(cost)
            if (batch_remainder > 0):
                # Do remainder
                k = -batch_remainder
                _x = np.matrix(x[k:]).T
                _y = np.matrix(y[k:])
                cost = self._train_step(sess, _x, _y)
                cost_list.append(cost)

        return cost_list
        

train_df = pd.read_csv(DATA_DIR + DATA_FILE, names=NAMES)
test_df = pd.read_csv(DATA_DIR + TEST_FILE, names=NAMES, skiprows=1)

training_data_dict = preprocess_data(train_df)
test_data_dict = preprocess_data(test_df,
                                 feature_list=training_data_dict['feature_list'],
                                 mean=training_data_dict['mean'],
                                 std=training_data_dict['std'])

# Create some parameter vectors for each feature set. Initialise to 0
num_features = training_data_dict['features'].shape[1]

# Train
train_features = training_data_dict['features']
train_target = training_data_dict['target']

test_features = test_data_dict['features']
test_target = test_data_dict['target']

num_epochs = 1000
batch_size = 3
learning_rate = 0.03

tf.reset_default_graph()
logreg = TFLogisticRegression(learning_rate, num_features, 1)
debug = True
with tf.Session() as sess:
    if debug: sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    logreg.initialise_graph(sess)
    
    cost = logreg.train(sess,
                        train_features,
                        train_target,
                        train_features.shape[0],
                        num_epochs)
    print("Initial cost:", cost[0])
    print("Final cost:", cost[-1])
    p = logreg.predict(sess, test_features.T)
    correct_test_predictions = np.logical_not(np.logical_xor(p > 0.5, test_target > 0.5))
    test_result_df = pd.DataFrame()
    test_result_df['predictions'] = pd.Series(p.flatten()>0.5)
    test_result_df['target'] = pd.Series(test_target.flatten() > 0.5)
    test_result_df['result'] = correct_test_predictions
    print("Success rate on test data:", test_result_df.result.sum()/len(test_result_df))
    