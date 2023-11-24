# Part 1

A mini-library used to build, train and test multi-layered neural networks.

### Public API:

- func `xavier_init(size, gain)` - Initialises weights for a network.

- class `MultiLayerNetwork(input_dim, neurons, activations)` -
  Builds a multi-layered network with the hidden layers of size specified by neurons and activation functions specified
  by activations.
    - `network.forward(x)` - performs a forward pass on the dataset _x_
    - `network(x)` - alias for network.forward(x)
    - `network.backward(x)` - performs a backwards pass on the dataset _x_
    - `network.update_params(learning_rate)` - performs one step of gradient descent with the given _learning rate_

- class `Trainer(network, batch_size, nb_epoch, learning_rate, loss_fun, shuffle_flag)` -
  Builds an object to train the neural network with the given hyper-parameters.
    - `trainer.train(input, target)` - trains the network using the hyper-parameters.
    - `trainer.eval_loss(input, target)` - evaluates the loss function for the data and returns a scalar.

- class `Preprocessor(data)` - Builds an object to preprocess datasets.
    - `preprocessor.apply(data)` - Performs preprocessing on the given dataset.
    - `preprocessor.revert(data)` - Reverts the preprocessing operations on the given dataset.

#### Example use - runs example_main(): _(assuming environment is correct)_

`python3 part1_nn_lib.py` - Will create an example network, preprocessor and trainer, train the network, evaluate the
loss then output the loss and accuracy of the network.

# Part 2

An optimal neural network designed to predict the price of houses in California using the California House Prices
Dataset.
We implemented this using PyTorch _(and our Part 1 mini-library, but the PyTorch implementation is more optimised, so we
submitted the PyTorch version)._

- class `Regressor(x, nb_epoch, no_hidden_layers, hidden_layer_size, learning_rate)` - builds and initalises the model
  with
  the given hyperparameters. Through hyperparameter tuning, we found these optimal
  hyperparameters: `nb_epoch = 1000, no_hidden_layers = 3, hidden_layer_size = 256, learning_rate = 0.05`.
    - `regressor._preprocessor(x, y, training)`- preprocesses the input datasets `x` and `y` (if provided)
    - `regressor.fit(x, y)` - train the network based on the input datasets `x` and `y` (if provided)
    - `regressor.predict(x)` - outputs the predicted value corresponding to the input `x`
    - `regressor.score(x, y)` - evaluate the model accuracy
- func `save_regressor(model)` - saves the trained model to `part2_model.pickle`
- func `load_regressor(model)` - load and return the trained model from `part2_model.pickle`
- func `RegressorHyperParameterSearch(x_train, y_train, x_validation, y_validation)` - performs a hyper-parameter for
  fine-tuning the regressor implemented in the Regressor class.

#### Example use - runs example_main(): _(assuming environment is correct)_

`python3 part2_house_value_regression.py` - Will load the data from `housing.csv`, shuffle it, train the regressor, then save the model to a `.pickle` file.


