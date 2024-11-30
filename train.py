import os
import torch

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from model import VGG19
from torch import optim
from loss import Style_loss, Content_Loss


def image_load(img_path):
    img = Image.open(img_path).convert('RGB')  # load Image
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # resize the to make them same size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the images
        ]
    )
    # apply transforms and add another dimension to the tensor
    img = image_transforms(img).unsqueeze(0)
    return img


# undo the normalization of the image to display it correctly
def image_save(tensor, path):
    dtype = tensor.dtype
    dev = tensor.device
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=dev)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=dev)

    image = tensor.mul(std[None, :, None, None]).add(mean[None, :, None, None])
    image = F.to_pil_image(image[0])
    image.save(path)


def training(epochs, alpha, beta, content_image, style_image):
    # copy of the content image with the gradients to optimize it's pixels

    model = VGG19()  # call the VGG model

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if torch.cuda.is_available():  # choose the device type we used GPU
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    content_image = content_image.to(device)
    style_image = style_image.to(device)

    model = model.to(device).eval()

    generated_image = content_image.clone().requires_grad_(True).to(device)

    optimizer = optim.LBFGS([generated_image])
    # optimizer = optim.Adam([generated_image], lr=0.01)

    content_features = model(content_image)
    style_features = model(style_image)

    for e in range(epochs):
        print(f"Staring epoch {e}/{epochs}")

        def closure():
            optimizer.zero_grad()
            generated_features = model(generated_image)

            style_ls = Style_loss(style_features, generated_features)
            cont_ls = Content_Loss(content_features, generated_features)
            total_loss = alpha * cont_ls + beta * style_ls

            total_loss.backward()
            if e % 25 == 0:
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_ls.item(), cont_ls.item()))

            return total_loss

        optimizer.step(closure)

        if e % 25 == 0:
            image_save(generated_image, f"ART4_{e}.png")

    image_save(generated_image, f"Art4_final.png")


def main():
    epochs = 100
    alpha = 1
    beta = 1000

    # base_path = "D:\Style Transfer"
    base_path = "/home/abdalla/Omar/Style Transfer"
    content_image = image_load(os.path.join(base_path, "mypic.jpeg"))
    style_image = image_load(os.path.join(base_path, "Starry_Night.jpg"))
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    training(epochs, alpha, beta, content_image, style_image)


if __name__ == "__main__":
    main()
