from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models


def image_load(content_image, style_image):
    img1 = Image.open(content_image).convert('RGB')  # load content Image
    img2 = Image.open(style_image).convert('RGB')  # load style Image

    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # resize the to make them same size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the images
        ]
    )  # add another dimension to the tensor
    return image_transforms(img1).unsqueeze(0), image_transforms(img2).unsqueeze(0)


def Gram_Matrix(style_features, generated_features):
    batch_size, channel, height, width = generated_features.size()  # get size of feature map
    Gen_matrix = generated_features.view(channel, height * width)  # vectorized features of Generated Image
    Style_matrix = style_features.view(channel, height * width)  # vectorized features of Style Image

    A = torch.mm(Style_matrix, Style_matrix.t())  # Gram Matrix of Style Representation layer
    G = torch.mm(Gen_matrix, Gen_matrix.t())  # Gram Matrix of Style feature layer

    return A, G


def Content_Loss(content_features, generated_features):
    # Mean Square Difference of content P &  generated F representation in one layer
    P = content_features['conv4_2']
    F = generated_features['conv4_2']

    cont_ls = torch.mean((P - F) ** 2)

    return cont_ls


def Style_loss(style_features, generated_features):
    style_ls = 0
    E = 0
    _, N, height, width = generated_features.size()  # get size of feature map

    style_weights = {'conv1_1': 0.2,
                     'conv2_1': 0.2,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}  # weighting factors of contribution of each layer
    for layer in style_weights:
        _, N, height, width = generated_features[layer].size()

        M = height * width

        A, G = Gram_Matrix(style_features[layer], generated_features[layer])  # compute gram for each layer

        E += torch.mean((G - A) ** 2) / (N * M)  # contribution of layers to the total loss

        style_ls += style_weights[layer] * E  # style loss of all layers

    return style_ls


def training(epochs, alpha, beta, content_image, style_image):
    # copy of the content image with the gradients to optimize it's pixels
    generated_image = content_image.clone().requires_grad_(True)

    model = VGG19()  # call the VGG19 model

    if torch.cuda.is_available():  # choose the device type we used GPU
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device).eval()

    optimizer = torch.optim.LBFGS([generated_image])

    for e in range(epochs):
        print(f"Staring epoch {e}/{epochs}")

        content_features = model(content_image.to(device))
        style_features = model(style_image.to(device))
        generated_features = model(generated_image.to(device))

        def closure():
            style_ls = Style_loss(style_features, generated_features)
            cont_ls = Content_Loss(content_features, generated_features)
            total_loss = alpha * cont_ls + beta * style_ls

            optimizer.zero_grad()
            total_loss.backward()

            return total_loss

        optimizer.step(closure)


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
