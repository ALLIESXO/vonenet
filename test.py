import torch, torchvision
import numpy
from vonenet import get_model
from PIL import Image
import matplotlib.pyplot as plt

model = get_model(controlled_params=True, pretrained=False)
voneblock = model.module

def torchNormalize(t):
    return (t - t.min()) / (t.max() - t.min())

image = Image.open("/home/alexander/Downloads/imagenet2012/ILSVRC2012_val_00000003.JPEG")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# store intermediate cropped image
torch_image = transforms(image)
intermediate = torchvision.transforms.ToPILImage()(torch_image[:3])
intermediate.save(f"intermediate.png", format="png")

torch_image = torch_image.unsqueeze(0)
result = voneblock(torch_image)[0]

normalized_result = (result - result.min()) / (result.max() - result.min())

for i in range(512):
    filt = torchvision.transforms.ToPILImage()(normalized_result[i])
    filt.save(f"filter{i}.png", format="png")
print("test")

first_kernel = voneblock.simple_conv_q0.weight.data[500]
first_kernel = torchNormalize(first_kernel)
torchvision.transforms.ToPILImage()(first_kernel).save("kernel0.png", format="png")