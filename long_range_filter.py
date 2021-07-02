import numpy as np 
from PIL import Image

alpha = 35
std = 5
r_max = 25

def deg_distance(alpha, beta):
        degrees = np.abs(beta-alpha)
        return degrees if degrees <= 180 else degrees - 360

def create_long_range_filter(orientation):

    B_rad = lambda r: np.exp((-0.5 * ((np.abs(r) - r_max)**2) / std**2)) if np.abs(r) > r_max else 1.0
    B_ang = lambda theta, phi: np.cos((theta - phi)*(2*np.pi / (2.0*alpha))) if np.abs(deg_distance(phi, theta)) <= alpha/2.0 else 0.0
    B_theta = lambda theta, phi, rad: B_ang(theta,phi) * B_rad(rad)

    kernel_size = 101
    phil = np.empty((kernel_size,kernel_size),dtype=object)
    filt = np.zeros((kernel_size,kernel_size))
    filt2 = np.zeros((kernel_size,kernel_size))
    center = int(kernel_size/2)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            
            y_pos = -i + center if i <= center else center - i
            x_pos = j - center

            phi = np.rad2deg(np.arctan2(y_pos, x_pos))
            phi = phi if phi <= 180 and phi > 0 else phi + 360
            
            r = np.hypot(x_pos, y_pos)
            
            if r < 1:
                filt[i,j] = 1
            else:
                phil[i,j] = (x_pos,y_pos, phi, r)
                if orientation < 90:
                    filt[i,j] = np.abs(B_theta(orientation + 180, phi, r))
                else:
                    filt[i,j] = np.abs(B_theta(orientation, phi, r))

            # combine both orientations
            filt2 = np.flip(filt)
            filt = np.maximum(filt, filt2)
            
    filt = filt / filt.max()
    return filt

for deg in range(0,190,10):
    filt1 = create_long_range_filter(deg)
    img = Image.fromarray((filt1*255).astype(np.uint8))
    img.save(f"long_range_filter/degree_{deg}.png")