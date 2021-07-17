
import numpy as np
import torch
import torchvision
import kornia
from torch import nn
from torch.nn import functional as F
from vonenet.utils import gabor_kernel
from vonenet.long_range_filter import create_long_range_filter
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase, ori_stride,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224, long_range_iterations=3):
        super().__init__()

        self.in_channels = 3

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size
        self.long_range_iterations = long_range_iterations

        # intermediate representations for testing
        self.v1_response = None

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None
        self.long_range_feedback = None

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase) # simple cells
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2) # complex cells 

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        
        
        self.combination = CombinationLayer()
        self.lrinteraction = LongRangeLayer(self.theta[:(int(ori_stride)*4)])

        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        self.v1_response = x.detach().clone()
        # Noise [Batch, out_channels, H/stride, W/stride]
        """
        iter = 0 
        for i in range(self.long_range_iterations):
            ######## TODO: Remove following test chunk of code
            image = x[0].clone().detach()
            # 0 batch size 1 channels 2 height 3 width  
            mean = image.mean(dim=(1,2)) + 0.00001
            std = image.std(dim=(1,2))   + 0.00001
            image = torchvision.transforms.Normalize(mean=mean,std=std)(image)

            
            if image.shape[0] > 3:
                for k in range(image.shape[0]):
                    torchvision.transforms.ToPILImage()(image[k]).save("/tmp/lr_checks/" + f"{k}.png")
            else:
                torchvision.transforms.ToPILImage()(image).save("/tmp/lr_checks/" + f"test.png")
            ######## END of chunk 

            x[:,256:] = self.combination(x[:,256:], self.long_range_feedback)
            self.long_range_feedback = self.lrinteraction(x[:,256:])
            iter += 1
        """
        
        x[:,256:] = self.combination(x[:,256:], self.long_range_feedback)
        x = F.instance_norm(x)
        
        self.v1_lr_response = x.detach().clone()

        self.long_range_feedback = None # reset after every iteration 

        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)

        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None


class CombinationLayer(nn.Module):
    """
    At the combination stage, feedforward complex cellresponsesCEand feedback
    long-range responses WE are added and subject to “shunting interaction,”
    i.e., a non-linear compression of high amplitude activity following the 
    Weber–Fechner law.
    """
    def __init__(self):
        super().__init__()

        # params from Hansen, Neumann paper
        self.delta_v = 2.0
        self.alpha_v = 0.2
        self.beta_v  = 10.0

    def forward(self, x, long_range_feedback):
        
        # first iteration W_theta is set to C_theta
        if long_range_feedback is None:
            w_theta = x.clone()
        else:
            w_theta = long_range_feedback.clone()

        net_theta = x + (self.delta_v * w_theta)
        result = self.beta_v * (net_theta/(self.alpha_v + net_theta))
        return result

from PIL import Image

class LongRangeLayer(nn.Module):
    def __init__(self, orientations, ksize=17, std=3, r_max=4, alpha=25):
        super().__init__()
        self.alpha = alpha # degrees
        self.std = std # std of the gaussian
        self.r_max = r_max # maximum 
        self.kernel_size = ksize
        self.ori_stride = len(orientations)
        self.lrfilter = []
        for ori in orientations:
            self.lrfilter.append(create_long_range_filter(ori,ksize,alpha,std,r_max))


    def forward(self, x):
        # parameters form Hansen, Neumann paper
        a_w = 0.2 # decay
        # scale factors 
        eta_p = 5.0
        eta_m = 2.0
        beta_w = 0.001

        gauss_surround = kornia.filters.GaussianBlur2d((7,7), (8.0,8.0))
        gauss_center = kornia.filters.get_gaussian_kernel1d(5, 0.5).unsqueeze(0).unsqueeze(0)

        #w_theta = x.clone()
        netp = x.clone() 
        netm = x.clone()
        orth_step = int(self.ori_stride/4)
        # excitatory input is provided by correlation of input and long range filter of same orientation
        for i in range(0,x.shape[1],self.ori_stride):
            for j in range(self.ori_stride):
                if j < self.ori_stride/2:
                    weights = torch.from_numpy(self.lrfilter[j]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    vals = F.relu((netp[:,i+j] - netp[:,i+j+orth_step])).unsqueeze(1)
                    vals_padded = F.pad(vals, (8,8,8,8), mode="replicate")
                    netp[:,i+j] = F.conv2d(vals_padded, weights, stride=1).squeeze(1)
                else:
                    weights = torch.from_numpy(self.lrfilter[j]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    vals = F.relu(netp[:,i+j] - netp[:,i+j-orth_step]).unsqueeze(1)
                    vals_padded = F.pad(vals, (8,8,8,8), mode="replicate")
                    netp[:,i+j] = F.conv2d(vals_padded, weights, stride=1).squeeze(1)

        # inhibitory effect by sampling of activity with isotropic gaussians
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                ### 1d conv with features isotropic gaussian from orientational

                # this variant runs the gaussian over all sizes and phases 
                # netm[:,i,j] = torch.from_numpy(gaussian_filter(netp[:,i,j], 0.5))

                # variant where we run gaussian over every orientation 
                for k in range(int(x.shape[1]/self.ori_stride)):
                    #netm[:,(k*8):(k+1)*8,i,j] = torch.from_numpy(gaussian_filter(netp[:,(k*8):(k+1)*8,i,j], 0.5))
                    padded_features = F.pad(netp[:,(k*8):(k+1)*8,i,j].unsqueeze(1), (2,2), mode="circular")
                    netm[:,(k*8):(k+1)*8,i,j] = F.conv1d(padded_features, gauss_center.type(torch.FloatTensor)).squeeze(1)
        
        # spatial gaussian with sigma 8 
        for i in range(x.shape[1]):
            #netm[:,i] = torch.from_numpy(gaussian_filter(netm[:,i], 8.0))
            netm[:,i] = gauss_surround(netm[:,i].unsqueeze(1)).squeeze(1)
        
        return beta_w * (x * (1.0 + eta_p * netp)/(a_w + eta_m * netm))
        
