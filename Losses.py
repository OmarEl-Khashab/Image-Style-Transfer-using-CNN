import torch
from torchvision.utils import save_image


def Gram_matrix(feature):
    _, N_filters, Height, Width = feature.size()  # get size of feature
    feature_tensor = feature.view(N_filters, Height * Width)  # vectorized features
    G = torch.mm(feature_tensor, feature_tensor.t())  # Compute Gram Matrix of features
    return G

def Style_loss(style_features, generated_features):
    style_ls = 0

    style_weights = {'conv1_1': 0.2,        #0.2
                     'conv2_1': 0.12,       #0.12
                     'conv3_1': 0.2,        #0.2
                     'conv4_1': 0.5,       #0.47
                     'conv5_1': 0.3}  # weighting factors of contribution of each layer  # 0.3

    for layer in style_weights:
        Gen = generated_features[layer]
        Style = style_features[str(layer)]

        G = Gram_matrix(Gen)
        A = Gram_matrix(Style)

        _, N, height, width = generated_features[str(layer)].size()  # get size of feature map
        M = height * width
        E = torch.mean((G - A) ** 2) / (N * M)  # contribution of layers to the total loss
        style_ls += style_weights[str(layer)] * E  # style loss of all layers

    return style_ls


def Content_Loss(content_features, generated_features):
    # Mean Square Difference of content 'P' &  generated 'F' representation in one layer
    P = content_features['conv4_2']
    F = generated_features['conv4_2']

    cont_ls = torch.mean((F - P) ** 2)

    return cont_ls
