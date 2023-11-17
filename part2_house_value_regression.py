import torch
import pickle
import numpy as np
import pandas as pd
from network import Network
from sklearn.preprocessing import LabelBinarizer, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch import nn

class Regressor():

    def __init__(self, x, nb_epoch = 1000, hidden_layers = [], learning_rate = 1e-2, momentum = 0.9):
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
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.momentum = momentum
        self.network = None
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

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        z = x.fillna(0)
        ct = ColumnTransformer ([
            ("num", MinMaxScaler(), ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"])
        ])
        resx = ct.fit_transform(z)

        resy = None
        if y is not None:
            m = y.fillna(0)
            ct2 = ColumnTransformer ([
                ("num", MinMaxScaler(), ["median_house_value"])
            ])
            resy = torch.from_numpy(ct2.fit_transform(m)) 
        return torch.from_numpy(resx), resy

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
        
        device = 'cpu'
        if torch.cuda.is_available():
            pass#device = 'cuda'
        
        #torch.randn((), device=device, dtype=torch.float64)
        x_train_tensor, y_train_tensor = self._preprocessor(x, y = y, training = True) # Do not forget
        x_train_tensor = x_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)

        # build a layer for each element in the hidden layer list
        input_features = len(x_train_tensor[0])
        output_features = 1
        layers_for_network = [input_features] + self.hidden_layers + [output_features]
        
        self.network = Network(layers_for_network).to(device)
        
        for epoch in range(self.nb_epoch):
            # Perform forward pass though the model given the input.
            run = self.network(x_train_tensor)
            # Compute the loss based on this forward pass.
            mse_loss = nn.MSELoss()
            result = mse_loss(run, y_train_tensor)
            # Perform backwards pass to compute gradients of loss with respect to parameters of the model.
            result.backward()
            # Perform one step of gradient descent on the model parameters.
            optimiser = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.momentum)
            optimiser.step()
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

        X, _ = self._preprocessor(x, training = False) # Do not forget
        with torch.no_grad():
            y_predicted = self.network(X)
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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_predicted = self.predict(x)
        # call some kind of evaluation function on y_predicted and Y
        # decide whether to micro or macro average these results
        
        return 0 # Replace this code with your own

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



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # tune hyperparams
    '''
    we have nb_epoch, hidden layer shape, learning_rate, and momentum as HPs
    '''
    return  # Return the chosen hyper parameters

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

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

