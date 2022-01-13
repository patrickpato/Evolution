import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from scipy.stats import boxcox 
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.linalg import pinv2, inv
import time
import numpy as np
import multiprocessing as mp
import pickle
from urllib.parse import urlparse
import logging
import sys

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

windows_10_df = pd.read_csv('windows10_dataset.csv')
features = windows_10_df.drop(columns=['ts', 'type'], axis=1)
numeric_features = features.select_dtypes(exclude=['object'])

## DATA TRANSFOROMATION
## Z-score transformation
def compute_z_score(df):
    z_score_df = df.copy()
    for col in df.columns:
        z_score_df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return z_score_df
## Min-Max transformation
def min_max_scaler(df):
    min_max_scaler_df = df.copy()
    for col in df.columns:
        min_max_scaler_df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
    return min_max_scaler_df
## Yeo-Johnson transformation
def Yeo_Johnson(df):
    yj_df = df.copy()
    yj = PowerTransformer(method='yeo-johnson')
    for col in df.columns:
        yj_df[col] = yj.fit_transform(df[[col]])
    return yj_df


z_transformed_df = compute_z_score(numeric_features)
scaled_df = min_max_scaler(numeric_features)
yj_df = Yeo_Johnson(numeric_features)

numeric_features['target'] = windows_10_df['type'].values
numeric_features = numeric_features.sample(frac=1)
sample_data = numeric_features.head(1000) # experimenting with the first 1000 random observations

le = LabelEncoder()
sample_data['target'] = le.fit_transform(sample_data['target'])

X = sample_data.drop(columns = ['target'], axis=1)
y = sample_data[['target']]
#Performing min_max scaling on the features
X_sc = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc,y,test_size=0.3, random_state=42)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

ga_weights = [0.23965973, 0.59444297, 0.15906395, 0.96999241, 0.49578476, 0.98579612,
   0.97379459, 0.10044159, 0.58018868, 0.98123659, 0.28536973, 0.10049513,
   0.59427999, 0.36067188, 0.18894792, 0.81008784, 0.30388576, 0.81563167,
   0.53892525, 0.8460454,  0.83077713, 0.77104079, 0.4005641,  0.39479354,
   0.67231298, 0.84327235, 0.50430458, 0.67486799, 0.25297094, 0.57516654,
   0.36880282, 0.94940265, 0.81269401, 0.40442764, 0.4552317,  0.74398229,
   0.56083737, 0.6648127,  0.76352565, 0.73914771, 0.95422556, 0.07842428,
   0.45819234, 0.6384561,  0.04039281, 0.35935401, 0.18226691, 0.09055939,
   0.50976097, 0.56777452, 0.01415905, 0.58033962, 0.09733806, 0.94514628,
   0.16804123, 0.32891374]

class ExtremeLearningMachine():
    '''
    Scratch implementation of Extreme Learning Machine
    -------------------
    Parameters:
    shape: list, shape[hidden units, output units]
        numbers of hidden units and output units
    activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh' or 'leaky_relu'
        Activation function to be used to convert y_hat to H
    x: array, shape[samples, features]
        train data/features
    y: array, shape[samples, ]
        target
    C: float
        regularization parameter
    elm_type: str, 'clf' or 'reg'
        'clf' means ELM solve classification problems, 'reg' means ELM solve regression problems.
    one_hot: bool, Ture or False, default True 
        The parameter is useful only when elm_type == 'clf'. If the labels need to transformed to
        one_hot, this parameter is set to be True
    random_type: str, 'uniform' or 'normal', default:'normal'
        Weight initialization method. Options: normal, uniform, EA
    '''
    def __init__(self, hidden_units, activation_function,  x, y, C, alg_type, one_hot=True, weight_randomization_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.weight_randomization_type = weight_randomization_type
        self.x = x
        self.y = y
        self.C = C
        self.class_num = np.unique(self.y).shape[0]     
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        self.alg_type = alg_type
        self.one_hot = one_hot

        # if classification problem and one_hot == True
        if alg_type == 'clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i, int(self.y[i])] = 1

        # Randomly generate the weight matrix and bias vector from input to hidden layer
        # 'uniform': uniform distribution
        # 'normal': normal distribution
        if self.weight_randomization_type == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
        if self.weight_randomization_type == 'normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))
        if self.weight_randomization_type == "GA":
            self.W = ga_weights
            self.b = np.random.normal(loc=0, scale=0.4, size=(self.hidden_units, 1))

    # compute the output of hidden layer according to different activation function
    def __input2hidden(self, x):
        self.yhat_tmp = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            self.H = 1/(1 + np.exp(- self.yhat_tmp))

        if self.activation_function == 'relu':
            self.H = self.yhat_tmp * (self.yhat_tmp > 0)

        if self.activation_function == 'sin':
            self.H = np.sin(self.yhat_tmp)

        if self.activation_function == 'tanh':
            self.H = (np.exp(self.yhat_tmp) - np.exp(-self.yhat_tmp))/(np.exp(self.yhat_tmp) + np.exp(-self.yhat_tmp))

        if self.activation_function == 'leaky_relu':
            self.H = np.maximum(0, self.yhat_tmp) + 0.1 * np.minimum(0, self.yhat_tmp)

        return self.H

    # compute the output
    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    '''
    Function: Train the model, compute beta matrix, the weight matrix from hidden layer to output layer
    ------------------
    Parameter:
    algorithm: str, 'no_re', 'solution1' or 'solution2'
        The algorithm to compute beta matrix
    ------------------
    Return:
    beta: array
        the weight matrix from hidden layer to output layer
    train_score: float
        the accuracy or RMSE
    train_time: str
        time of computing beta
    '''
    def fit(self, algorithm):
        self.time1 = time.clock()   # compute running time
        self.H = self.__input2hidden(self.x)
        if self.alg_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        if self.alg_type == 'reg':
            self.y_temp = self.y
        # no regularization
        if algorithm == 'no_re':
            self.beta = np.dot(pinv2(self.H.T), self.y_temp)
        # faster algorithm 1
        if algorithm == 'solution1':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.tmp1, self.H)
            self.beta = np.dot(self.tmp2, self.y_temp)
        # faster algorithm 2
        if algorithm == 'solution2':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.H.T, self.tmp1)
            self.beta = np.dot(self.tmp2.T, self.y_temp)
        self.time2 = time.clock()

        # compute the results
        self.result = self.__hidden2output(self.H)
        # If the problem if classification problem, the output is softmax
        if self.alg_type == 'clf':
            self.result = np.exp(self.result)/np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

        # Evaluate training results
        # If problem is classification, compute the accuracy
        # If problem is regression, compute the RMSE
        if self.alg_type == 'clf':
            self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
            self.correct = 0
            for i in range(self.y.shape[0]):
                if self.y_[i] == self.y[i]:
                    self.correct += 1
            
            self.train_score = self.correct/self.y.shape[0]
        if self.alg_type == 'reg':
            self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y))/self.y.shape[0])
        train_time = str(self.time2 - self.time1)
        return self.beta, self.train_score, train_time

    '''
    Function: compute the result given data
    ---------------
    Parameters:
    x: array, shape[samples, features]
    ---------------
    Return:
    y_: array
        predicted results
    '''
    def predict(self, x):
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        if self.alg_type == 'clf':
            self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

        return self.y_

    '''
    Function: compute accuracy or RMSE given data and labels
    -------------
    Parameters:
    x: array, shape[samples, features]
    y: array, shape[samples, ]
    -------------
    Return:
    test_score: float, accuracy or RMSE
    '''
    def score(self, x, y):
        self.prediction = self.predict(x)
        if self.alg_type == 'clf':
            self.correct = 0
            for i in range(y.shape[0]):
                if self.prediction[i] == y[i]:
                    self.correct += 1
            self.test_score = self.correct/y.shape[0]
        if self.alg_type == 'reg':
            self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y))/self.y.shape[0])

        return self.test_score
    ## For evolutionary strategies
    def get_weights(self):
        return self.W

    def set_weights(self, weights):
        self.weights = self.W

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
#setting parameters
act_funct  = sys.argv[1] if len(sys.argv) > 1 else "relu" # running with relu activation function
randomization_method = sys.argv[2] if len(sys.argv) > 1 else "normal" #running with normal weight randomization method. 
with mlflow.start_run():
        elm1 = ExtremeLearningMachine(hidden_units=32, activation_function=act_funct, weight_randomization_type=randomization_method, x=X_train, y=y_train, C=0.1, alg_type='clf')
        #lr.fit(train_x, train_y)

        #predicted_qualities = lr.predict(test_x)

        beta, train_accuracy, running_time = elm1.fit('solution2')
        predictions_sample = elm1.predict(X_test)
        #clf_report = confusion_matrix(predictions_sample, X_test)


        print("ELM model accuracy:" +str(train_accuracy))
        
       

        mlflow.log_param("activation_function", act_funct)
        mlflow.log_param("weight_randomization_method", randomization_method)
        mlflow.log_metric("Accuracy", train_accuracy)
        
        #mlflow.log_metric("r2", r2)
        #mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(elm1, "model", registered_model_name="ELM_v1")
        else:
            mlflow.sklearn.log_model(elm1, "model")