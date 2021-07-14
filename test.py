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

stimulus = Image.open("/home/alexander/vipp/vonenet/LiCircleStim.png")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5),
])

# test long range filter 
stimulus_torch = transforms(stimulus).repeat(3,1,1).unsqueeze(0)
"""
weights = torch.from_numpy(voneblock.lrinteraction.lrfilter[1]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
stimulus_torch = torch.nn.functional.conv2d(stimulus_torch, weights)
torchvision.transforms.ToPILImage()(stimulus_torch[0]).save(f"./stimulus.png")
"""
stimulus_LR = voneblock(stimulus_torch)[0]
stimulus_v1 = voneblock.v1_response[0]
torchvision.transforms.ToPILImage()(stimulus_torch[0]).save(f"./stimulus_center_cropped.png")
for i in range(512):
    torchvision.transforms.ToPILImage()(stimulus_v1[i]).save(f"./stim_gfb/stimulus{i}.png")
    torchvision.transforms.ToPILImage()(stimulus_LR[i]).save(f"./stim_lr/stimulus{i}.png")
    
# store intermediate cropped image
torch_image = transforms(image)
intermediate = torchvision.transforms.ToPILImage()(torch_image[:3])
intermediate.save(f"intermediate.png", format="png")

torch_image = torch_image.unsqueeze(0)
result = voneblock(torch_image)[0]

test = result[:8].max(dim=0).values
torchvision.transforms.ToPILImage()(test).save(f"./test.png", format="png")
normalized_result = (result - result.min()) / (result.max() - result.min())

for i in range(512):
    filt = torchvision.transforms.ToPILImage()(normalized_result[i])
    filt.save(f"./filter_outputs/filter{i}.png", format="png")
    kernel = voneblock.simple_conv_q0.weight.data[i]
    kernel = torchNormalize(kernel)
    torchvision.transforms.ToPILImage()(kernel).save(f"./kernels/kernel{i}.png", format="png")

print("Done!")
