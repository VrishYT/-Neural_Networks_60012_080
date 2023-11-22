import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import part1_nn_lib as nn


class Regressor():

    def __init__(self,
                 x: pd.DataFrame,
                 nb_epoch: int = 1000,
                 learning_rate: float = 1e-2,
                 shuffle: bool = False,
                 batch_size: int = 100,
                 output_size: int = 1):
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
        self.nb_epoch: int = nb_epoch
        self.preprocessor = None
        pro_x, _ = self._preprocessor(x, training=True)

        self.input_size = pro_x.shape[1]
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.stored_classes = None

        self.network = nn.MultiLayerNetwork(self.input_size, [128, 64, output_size], ["relu"] * 3)
        self.trainer = nn.Trainer(
            self.network,
            batch_size,
            nb_epoch,
            learning_rate,
            "mse",
            shuffle
        )

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

        if training:
            self.preprocessor = nn.Preprocessor(numerical_cols)

        preprocessed_x = self.preprocessor.apply(numerical_cols)

        result_x = np.concatenate((preprocessed_x, one_hot_encoded_data), axis=1)

        if y is not None:
            y = y.to_numpy()

        return result_x, y

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
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        self.trainer.train(X, Y)

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
        return self.network(X)

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
        result = np.sqrt(self.trainer.eval_loss(X, Y))

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


def RegressorHyperParameterSearch(xTrain, yTrain, xValidate, yValidate, mins, maxs, steps):
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

    # currentLoss = False
    # currentParams = None
    # print("jashtye")
    # for nb_epoch in range(mins['nb_epoch'], maxs['nb_epoch'], steps['nb_epoch']):
    #     print("ne epoch")
    #     for learning_rate in range(mins['learning_rate'], maxs['learning_rate']):
    #         print("learn")
    #         learning_rate *= steps['learning_rate']
    #         for shuffle in [True, False]:
    #             print("shuf")
    #             for batch_size in range(mins['batch_size'], maxs['batch_size'], steps['batch_size']):
    #                 print("batchi")
    #                 regressor = Regressor(xTrain, nb_epoch=nb_epoch,
    #                                         learning_rate=learning_rate,
    #                                         batch_size=batch_size, shuffle=shuffle)
    #                 print("para palestres")
    #                 regressor.fit(xTrain, yTrain)

    #                 print("pas palestres")
    #                 # calculate loss of regressor
    #                 loss = regressor.score(xValidate, yValidate)

    #                 print("loosi", loss)
    #                 print("se ca mutin", {
    #                         'nb_epoch': nb_epoch,
    #                         'batch_size': batch_size,
    #                         'learning_rate': learning_rate,
    #                         'shuffle': shuffle
    #                     })
    #                 if (currentLoss == False) or (loss < currentLoss):
    #                     currentLoss = loss
    #                     currentParams = {
    #                         'nb_epoch': nb_epoch,
    #                         'batch_size': batch_size,
    #                         'learning_rate': learning_rate,
    #                         'shuffle': shuffle
    #                     }
    #                 print("current loss ", currentLoss)
    #                 print("current params ", currentParams)
    currentLoss = False
    currentParams = None

    for nb_epoch in range(mins['nb_epoch'], maxs['nb_epoch'], steps['nb_epoch']):
        for learning_rate in range(mins['learning_rate'], maxs['learning_rate']):
            learning_rate *= steps['learning_rate']
            for nr_hidden_layers in range(mins['nr_hidden_layers'], maxs['nr_hidden_layers'],
                                          steps['nr_hidden_layers']):
                for hidden_layer_size in range(mins['hidden_layer_size'], maxs['hidden_layer_size']):
                    hidden_layer_size *= steps['hidden_layer_size']
                    for activation_function in ["relu", "sigmoid"]:

                        regressor = Regressor(xTrain, nb_epoch=nb_epoch,
                                              learning_rate=learning_rate,
                                              )
                        regressor.network = nn.MultiLayerNetwork(regressor.input_size,
                                                                 [hidden_layer_size] * int(nr_hidden_layers),
                                                                 [activation_function] * int(nr_hidden_layers))
                        regressor.fit(xTrain, yTrain)

                        # calculate loss of regressor
                        loss = regressor.score(xValidate, yValidate)

                        print("loosi", loss)
                        print("se ca mutin", {
                            'nb_epoch': nb_epoch,
                            'hidden_layer_size': hidden_layer_size,
                            'learning_rate': learning_rate,
                            'nr_hidden_layers': nr_hidden_layers,
                            'activation_function': activation_function
                        })
                        if (currentLoss == False) or (loss < currentLoss):
                            currentLoss = loss
                            currentParams = {
                                'nb_epoch': nb_epoch,
                                'hidden_layer_size': hidden_layer_size,
                                'learning_rate': learning_rate,
                                'nr_hidden_layers': nr_hidden_layers,
                                'activation_function': activation_function
                            }
                        print("current loss ", currentLoss)
                        print("current params ", currentParams)

    return currentParams  # Return the chosen hyper parameters

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

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

    # HP Tune
    mins = {'nb_epoch': 1000, 'batch_size': 1, 'learning_rate': 1, 'nr_hidden_layers': 1, 'hidden_layer_size': 32}
    maxs = {'nb_epoch': 1500, 'batch_size': 3, 'learning_rate': 6, 'nr_hidden_layers': 4, 'hidden_layer_size': 1024}
    steps = {'nb_epoch': 50, 'batch_size': 1, 'learning_rate': 5e-3, 'nr_hidden_layers': 1, 'hidden_layer_size': 2}
    # print("po hym")
    best = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test, mins, maxs, steps)
    # print("dolem nga tuneli")

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=best['nb_epoch'], batch_size=best['batch_size'],
                          learning_rate=['learning_rate'])
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
