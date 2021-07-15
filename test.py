import torch, torchvision
import numpy
from vonenet import get_model
from PIL import Image
import kornia

# global values / parameters 
gauss_surround = kornia.filters.GaussianBlur2d((7,7), (8.0,8.0))
model = get_model(controlled_params=True, pretrained=False)
voneblock = model.module
dataset_test_image = "R:/Datasets/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG"
stimulus = "R:/Datasets/LiCircleStim.png"

def tNormalize(t):
    return (t - t.min()) / (t.max() - t.min())

def load_image_via_transformation(image_path):
    # loads image, resizes, normalizes and returns as an rgb image with batch size 1 
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5),
    ])
    image = Image.open(image_path)
    image = transforms(image)
    # turn grey images to rgb
    if image.shape[0] == 1:
        image = image.repeat(3,1,1)
    
    return image.unsqueeze(0) 


def save_image_normalized(image, path, name="default"):
    image = torchvision.transforms.Normalize(mean=image.mean(), std=image.std())(image)[0]
    if image.shape[0] > 3:
        for i in range(image.shape[0]):
            torchvision.transforms.ToPILImage()(image[i]).save(path + f"{name}{i}.png")
    else:
        torchvision.transforms.ToPILImage()(image).save(path + f"{name}.png")


if __name__ == '__main__':
    
    stim_torch = load_image_via_transformation(stimulus)
    result = voneblock(stim_torch)
    save_image_normalized(result, "R:/VOneBlockResults/")
    image = voneblock.v1_response
    lr_image = voneblock.v1_lr_response
    save_image_normalized(lr_image, "R:/LR_Responses/")
    image = torchvision.transforms.Normalize(mean=image.mean(), std=image.std())(image)[0]
    for i in range(image.shape[0]):
        torchvision.transforms.ToPILImage()(image[i]).save( f"R:/V1_Responses/{i}.png")
    print("Done!")



"""
for i in range(8):
    weights = torch.from_numpy(voneblock.lrinteraction.lrfilter[i]).type(torch.float).unsqueeze(0).unsqueeze(0)
    filtered_stim = torch.nn.functional.conv2d(stimulus_torch[:,1,:,:].unsqueeze(1), weights)
    torchvision.transforms.ToPILImage()(filtered_stim[0]).save(f"./{i}stimulus.png")


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
"""