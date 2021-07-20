import torch, torchvision
import numpy
from vonenet import get_model
from PIL import Image
import kornia
import sys
is_windows = hasattr(sys, 'getwindowsversion')
# global values / parameters 
gauss_surround = kornia.filters.GaussianBlur2d((7,7), (8.0,8.0))
model = get_model(controlled_params=True, pretrained=False)
voneblock = model.module
if is_windows:
    dataset_test_image = "R:/Datasets/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG"
    stimulus = "R:/Datasets/LiCircleStim.png"
    store_path = "R:"
else: 
    dataset_test_image = "/mnt/M2SSD/Datasets/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG"
    stimulus = "/mnt/M2SSD/Datasets/LiCircleStim.png"
    store_path = "/mnt/M2SSD"

# very simple stimulis 
simple_bar_white = "/mnt/M2SSD/Datasets/brbar_simple_.png"
simple_circle_black = "/mnt/M2SSD/Datasets/circle_simple_black.png"

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
    
    stim_torch = load_image_via_transformation(simple_bar_white)
    stim_torch = stim_torch[:,:3,:,:]
    result = voneblock(stim_torch)
    save_image_normalized(result, f"{store_path}/VOneBlockResults/")
    image = voneblock.v1_response
    lr_image = voneblock.v1_lr_response
    save_image_normalized(lr_image * 100, f"{store_path}/LR_Responses/")
    image = torchvision.transforms.Normalize(mean=image.mean(), std=image.std())(image)[0]
    for i in range(image.shape[0]):
        torchvision.transforms.ToPILImage()(image[i]).save( f"{store_path}/V1_Responses/{i}.png")
    print("Done!")

