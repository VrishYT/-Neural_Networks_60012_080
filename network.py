from torch import nn

class Network(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.hidden_layer_shape = hidden_layers
        self.layers = self._hidden_layer_generator(hidden_layers)
        self.double()
        
    def _hidden_layer_generator(self, hidden_layers):
        num_hidden_layers = len(hidden_layers)
        layers = []
        for i in range(num_hidden_layers-1):
            if i > 0:
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(in_features=hidden_layers[i], out_features=hidden_layers[i+1]))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)