import torch
import pickle
import numpy as np
import pandas as pd
from network import Network
from sklearn.preprocessing import LabelBinarizer, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from torch import nn


class Regressor():

    def __init__(self, x, nb_epoch=1500, hidden_layers=[14, 16, 7], learning_rate=0.1):
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
        self.minMax = MinMaxScaler()
        self.ct = ColumnTransformer([
            ("num", self.minMax,
             ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
              "households", "median_income"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])
        ])
        X = self._preprocessor(x, training=True)
        self.input_size: int = X.shape[1]
        self.output_size: int = 1
        self.nb_epoch: int = nb_epoch
        self.learning_rate: float = learning_rate
        self.hidden_layers = hidden_layers
        print("hidden layers", self.hidden_layers)
        layers_for_network = [self.input_size] + self.hidden_layers + [self.output_size]
        self.network = Network(layers_for_network)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x: pd.DataFrame, y: pd.DataFrame = None, training: bool = False):
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

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        defaults = {"longitude": x["longitude"].mean(),
                    "latitude": x["latitude"].mean(),
                    "housing_median_age": x["housing_median_age"].mean(),
                    "total_rooms": x["total_rooms"].mean(),
                    "total_bedrooms": x["total_bedrooms"].mean(),
                    "population" : x["population"].mean(),
                    "households" : x["households"].mean(),
                    "median_income" : x["median_income"].mean(),
                    "ocean_proximity" : x["ocean_proximity"].mode()[0]}
        
        z = x.fillna(value=defaults)

        if training:
            self.ct = self.ct.fit(z)

        resx = self.ct.transform(z)

        resy = None
        if y is not None:
            # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            y = y.fillna(y["median_house_value"].mean())
            #m = self.minMax.fit_transform(y)
            resy = torch.from_numpy(y.to_numpy())
            print('resy', resy)
        print('resx', resx)
        return torch.from_numpy(resx)

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

        x_train_tensor = self._preprocessor(x, y=y, training=True)  # Do not forget
        y_train_tensor = torch.tensor(y.values, dtype=torch.float64)

        # build a layer for each element in the hidden layer list

        for epoch in range(self.nb_epoch):
            # Perform forward pass though the model given the input.
            # print("x_train_tensor", x_train_tensor.size())
            run = self.network(x_train_tensor)
            # Compute the loss based on this forward pass.
            mse_loss = nn.MSELoss()
            result = mse_loss(run, y_train_tensor)
            # Perform backwards pass to compute gradients of loss with respect to parameters of the model.
            result.backward()
            # Perform one step of gradient descent on the model parameters.
            self.optimiser.step()
            # You are free to implement any additional steps to improve learning (batch-learning, shuffling...).

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

        X = self._preprocessor(x, training=False)  # Do not forget
        with torch.no_grad():
            y_predicted = self.network(X)
        print(y_predicted)
        return y_predicted

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

        #_, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        y_predicted = self.predict(x)
        # call some kind of evaluation function on y_predicted and Y
        
        #y_predicted = self.minMax.inverse_transform(y_predicted)
        print('posttransform', y_predicted)
        result = self.loss_func(y_predicted, torch.from_numpy(y.to_numpy()))

        return np.sqrt(result)  # Replace this code with your own

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


def RegressorHyperParameterSearch(xTrain, yTrain, xValidate, yValidate, mins, maxs, steps):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    // TODO: add typedef for Params type - https://stackoverflow.com/questions/69446189/python-equivalent-for-typedef

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - mins {dict[str, float]} -- Minimum values for each HP:
            nb_epoch, nb_hidden_layers, hidden_layer_size, learning_rate
        - maxs {dict[str, float]} -- Maximum values for each HP:
            nb_epoch, nb_hidden_layers, hidden_layer_size, learning_rate
        - steps {dict[str, float]} -- Step sizes for each HP:
            nb_epoch, nb_hidden_layers, hidden_layer_size, learning_rate
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # tune hyperparams
    '''
    We have nb_epoch, hidden layer shape and learning_rate as HPs
    '''

    def genList(n, min, max, step):
        possibilities = (max - min) / step
        array = np.zeros((int(possibilities), int(n)), dtype=int)

        for i in range(n):
            for j, hidden_layer_size in enumerate(range(min, max, step)):
                array[j][i] = hidden_layer_size

        print('array', array)
        return array.tolist()

    # check nb_epoch
    currentLoss = False
    currentParams = None

    for nb_epoch in range(mins['nb_epoch'], maxs['nb_epoch'], steps['nb_epoch']):
        for nb_hidden_layers in range(mins['nb_hidden_layers'], maxs['nb_hidden_layers'], steps['nb_hidden_layers']):
            array = genList(nb_hidden_layers, mins['hidden_layer_size'], maxs['hidden_layer_size'],
                            steps['hidden_layer_size'])
            for hidden_layer_shape in array:
                for learning_rate in range(mins['learning_rate'], maxs['learning_rate']):
                    learning_rate *= steps['learning_rate']
                    regressor = Regressor(xTrain, nb_epoch=nb_epoch, hidden_layers=hidden_layer_shape,
                                          learning_rate=learning_rate)
                    regressor.fit(xTrain, yTrain)
                    # calculate loss of regressor
                    loss = regressor.score(xValidate, yValidate)
                    if (currentLoss == False) or (loss < currentLoss):
                        currentLoss = loss
                        currentParams = {
                            'nb_epoch': nb_epoch,
                            'hidden_layers': hidden_layer_shape,
                            'learning_rate': learning_rate
                        }

    return currentParams  # Return the chosen hyper parameters

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
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)


    # nb_epoch, nb_hidden_layers, hidden_layer_size, learning_rate
    mins = {'nb_epoch': 1000, 'nb_hidden_layers': 2, 'hidden_layer_size': 14, 'learning_rate': 1}
    maxs = {'nb_epoch': 1005, 'nb_hidden_layers': 6, 'hidden_layer_size': 20, 'learning_rate': 4}
    steps = {'nb_epoch': 1, 'nb_hidden_layers': 1, 'hidden_layer_size': 1, 'learning_rate': 1e-2}


    # best = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test, mins, maxs, steps)
    # print('best', best)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    # regressor = Regressor(x_train, nb_epoch=best['nb_epoch'], hidden_layers=best['hidden_layers'], learning_rate=best['learning_rate'])
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
