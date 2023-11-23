import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from part1_nn_lib import MultiLayerNetwork, Trainer
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import math

class Regressor():

    def __init__(self, x, nb_epoch = 100, no_hidden_layers=2, hidden_layer_size=256, learning_rate=0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.model = None
        self.stored_classes = None
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1

        self.nb_epoch = nb_epoch 
        self.no_hidden_layers = no_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        numerical_columns = x.select_dtypes(include=np.number).columns
        categorical_columns = x.select_dtypes(include="object").columns
        default_vals = {}
        for column in numerical_columns:
            default_vals[column] = x[column].mean()
        for column in categorical_columns:
            default_vals[column] = x[column].mode()[0]
        
        x = x.fillna(value=default_vals)
        
        fit_lb = None
        if training:
            lb = LabelBinarizer()
            fit_lb = lb.fit(categorical_columns)
            self.stored_classes=fit_lb.classes_
        else:
            fit_lb = LabelBinarizer()
            fit_lb.classes_ = self.stored_classes

        one_hot_encoded_data = fit_lb.transform(categorical_columns)
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=fit_lb.classes_)
        x = pd.concat([x, one_hot_encoded_df], axis=1)
        x.drop(columns=categorical_columns, axis=1, inplace=True)
        
        scaler = MinMaxScaler()
        x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

        return torch.tensor(x.values, dtype=torch.float32), (torch.tensor(y.values, dtype=torch.float32) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True)

        layers = OrderedDict([("input_layer",nn.Linear(self.input_size, self.hidden_layer_size, bias=True))])

        for i in range(self.no_hidden_layers):
            hidden_layer_name = "hidden_layer" + str(i+1) 
            relu_name = "relu" + str(i+1)
            layers[hidden_layer_name] = nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=True)
            layers[relu_name] = nn.ReLU()

        layers["output_layer"] = nn.Linear(self.hidden_layer_size,self.output_size, bias=True)

        self.model = nn.Sequential(layers)
        loss_fn = nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for n in range(self.nb_epoch):
            y_pred = self.model(X.float())
            loss = loss_fn(y_pred, Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        return self.model(X).detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = False)
        return (mean_squared_error(self.model(X).detach().numpy(), Y.numpy()))**0.5

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x_train, y_train, x_validation, y_validation):
    #######################################################################
    #                       ** START OF YOUR CODE **
    ####################################################################### 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 
        Returns a tuple of optimal parameters (number of hiddden layers, size of hidden layers, learning rate, number of epochs)
    """
    
    potential_num_hl = [2]
    potential_size_hl = [2,4,8,10,16,32,64,128,256,512]
    potential_lrs = [0.001, 0.005, 0.01, 0.05, 0.1]

    min_score = float('inf')
    best_parameters = None

    for i in potential_num_hl:
        for j in potential_size_hl:
            for k in potential_lrs:
                for l in [10**ll for ll in range(1,4)]:
                    regressor = Regressor(x_train,no_hidden_layers=i,hidden_layer_size=j, learning_rate=k, nb_epoch=l)
                    regressor.fit(x_train, y_train)
                    curr_score = regressor.score(x_validation, y_validation)
                    print("params", (i,j,k,l), "score", curr_score)

                    if curr_score < min_score:
                        min_score = curr_score
                        best_parameters = (i,j,k,l)

    return best_parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    
    shuffled_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffled_indices)
    # Splitting input and output

    shuffled_data = data.iloc[shuffled_indices]
    #shuffle

    x = shuffled_data.loc[:, data.columns != output_label]
    y = shuffled_data.loc[:, [output_label]]

    n_train = int(len(shuffled_indices)*0.8)

    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    x_test = x.iloc[n_train:]
    y_test = y.iloc[n_train:]

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    (optimal_num_hidden_layers, optimal_hidden_size, optimal_lr, optimal_epoch) = example_main_hyper_parameters()
    regressor = Regressor(x_train, nb_epoch=optimal_epoch, no_hidden_layers=optimal_num_hidden_layers, hidden_layer_size=optimal_hidden_size, learning_rate=optimal_lr)
    #regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)

    save_regressor(regressor)
    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

def example_main_hyper_parameters():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    
    shuffled_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffled_indices)
    # Splitting input and output

    shuffled_data = data.iloc[shuffled_indices]
    #shuffle

    x = shuffled_data.loc[:, data.columns != output_label]
    y = shuffled_data.loc[:, [output_label]]

    n_train = int(len(shuffled_indices)*0.8)

    x_train = x.iloc[:n_train]
    y_train = y.iloc[:n_train]

    x_validation = x.iloc[n_train:]
    y_validation = y.iloc[n_train:]

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    x_validation = x_validation.reset_index(drop=True)
    y_validation = y_validation.reset_index(drop=True)

    (optimal_num_hidden_layers, optimal_hidden_size, optimal_lr, optimal_epoch) = RegressorHyperParameterSearch(x_train, y_train, x_validation, y_validation)

    print("BEST")
    print(optimal_num_hidden_layers)
    print(optimal_hidden_size)
    print(optimal_lr)
    print(optimal_epoch)

    return (optimal_num_hidden_layers, optimal_hidden_size, optimal_lr, optimal_epoch)

if __name__ == "__main__":
    example_main()
