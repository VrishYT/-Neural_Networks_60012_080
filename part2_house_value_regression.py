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

    def __init__(self,
                 x: pd.DataFrame,
                 nb_epoch: int = 100,
                 learning_rate: float = 0.1,
                 no_hidden_layers: int = 2,
                 hidden_layer_size: int = 128,
                 ):
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

        # Replace this code with your own
        self.model = None
        self.stored_classes = None
        pro_x, _ = self._preprocessor(x, training=True)
        self.input_size = pro_x.shape[1]
        self.output_size = 1
        

        
        self.nb_epoch: int = nb_epoch
        self.no_hidden_layers = no_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        return 
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training: bool = False):

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
        def fill_missing_nums(*datas):
            for data in datas:
                if data is not None:
                    for col in data.columns:
                        data.loc[:, col] = data[col].fillna(data[col].mean())

        numerical_cols = x.select_dtypes(include='number')
        categorical_cols = x.select_dtypes(include='object')

        # Fill missing data
        fill_missing_nums(numerical_cols, y)
        for col in categorical_cols.columns:
            categorical_cols.loc[:, col] = categorical_cols[col].fillna(categorical_cols[col].mode()[0])

        fit_lb = None
        if training:
            lb = LabelBinarizer()
            fit_lb = lb.fit(categorical_cols)
            self.stored_classes = fit_lb.classes_
        else:
            fit_lb = LabelBinarizer()
            fit_lb.classes_ = self.stored_classes

        one_hot_encoded_data = fit_lb.transform(categorical_cols)

        scaler = MinMaxScaler()

        preprocessed_x = scaler.fit_transform(numerical_cols)

        result_x = np.concatenate((preprocessed_x, one_hot_encoded_data), axis=1)

        if y is not None:
            y = torch.tensor(y.values, dtype=torch.float32)

        return torch.Tensor(result_x), y

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

        layers = OrderedDict([("input_layer",nn.Linear(self.input_size, self.hidden_layer_size))])

        for i in range(self.no_hidden_layers):
            hidden_layer_name = "hidden_layer" + str(i+1) 
            layers[hidden_layer_name] = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)

        layers["output_layer"] = nn.Linear(self.hidden_layer_size,self.output_size)
        layers["output_layer_act"] = nn.ReLU()

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
        """                    regressor.fit(xTrain, yTrain)

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

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        return self.model(X).detach().numpy

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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        result = (mean_squared_error(self.model(X).detach().numpy(), Y.numpy()))**0.5

        return result

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


def RegressorHyperParameterSearch(xTrain, yTrain, xValidate, yValidate):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 
        nb_epoch: int = 1000,
        learning_rate: float = 1e-2,
        shuffle: bool = False,
        batch_size: int = 1

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    minLoss = False
    minParams = None

    for nb_epoch in [10**i for i in range(1,4)]:
        for learning_rate in [10**(-i) for i in range(1,6)]:
            for nr_hidden_layers in range(1,5):
                for hidden_layer_size in [2**i for i in range(1,11)]:
                    # for activation_function in ["relu", "sigmoid"]:
                    #     for shuffle in [True, False]:

                            activation_function = "relu"
                            shuffle = False

                            currentParams = {
                                'nb_epoch': nb_epoch,
                                'learning_rate': learning_rate,
                                'nr_hidden_layers': nr_hidden_layers,
                                'hidden_layer_size': hidden_layer_size,
                                'activation_function': activation_function,
                                'shuffle': shuffle
                            }

                            print("Current:", currentParams)

                            regressor = Regressor(xTrain,
                                                  nb_epoch=nb_epoch,
                                                  learning_rate=learning_rate,
                                                  shuffle=shuffle,
                                                  activation_function=activation_function,
                                                  hidden_layer_size=hidden_layer_size
                                                  )

                            regressor.fit(xTrain, yTrain)

                            # calculate loss of regressor
                            loss = regressor.score(xValidate, yValidate)
                            print("loss =", loss)
                            if (minLoss is False) or (loss < minLoss):
                                minLoss = loss
                                minParams = currentParams
                            print("current min loss", minLoss)

    return minParams  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    #shuffle
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:,[output_label]]

    validation_start = (data.shape[0] // 10) * 2
    validation_end = (data.shape[0] // 10)

    x_train = x.iloc[:-validation_start]
    y_train = y.iloc[:-validation_start]

    x_validation = x.iloc[-validation_start:-validation_end]
    y_validation = y.iloc[-validation_start:-validation_end]
    x_validation = x_validation.reset_index(drop=True)
    y_validation = y_validation.reset_index(drop=True)

    x_test = x.iloc[-validation_end:]
    y_test = y.iloc[-validation_end:]
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
