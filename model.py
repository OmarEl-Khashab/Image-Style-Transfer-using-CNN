import torch
import torch.nn as nn
import torchvision.models as models


# Defining a class that for the model
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.req_layers = {'0': 'conv1_1',
                           '5': 'conv2_1',
                           '10': 'conv3_1',
                           '19': 'conv4_1',
                           '21': 'conv4_2',  # content layer as the paper
                           '28': 'conv5_1'}

        self.model = models.vgg19(pretrained=True).features

        for param in self.model.parameters():
            param.requires_grad_(False)

        # replacing max pooling by average pooling yields slightly more appealing results
        for i, layer in enumerate(self.model):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    # x holds the input tensor(image) that will be fed to each layer
    def forward(self, x):
        # initialize an array t hat wil hold the activations from the chosen layers
        features = {}
        # Iterate over all the layers of the mode
        for name, layer in enumerate(self.model):
            x = layer(x)
            # appending the activation of the selected layers and return the feature array
            if str(name) in self.req_layers:
                features[self.req_layers[str(name)]] = x  # adding features to key values ( name of layer )
        return features
