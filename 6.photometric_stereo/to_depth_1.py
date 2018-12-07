## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):
    size = mask.shape
    mask = np.where(mask)
    fx, fy, fr = np.zeros(size), np.zeros(size), np.zeros(size)
    grid = np.meshgrid(range(-1,2),range(-1,2),indexing='ij')
    fx[grid] = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    fy[grid] = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    fr[grid] = np.array([[-1/9, -1/9, -1/9], [-1/9, 8/9, -1/9], [-1/9, -1/9, -1/9]])
    gx, gy = np.zeros(size), np.zeros(size)
    gx[mask] = -nrm[mask][:, 0] / np.maximum(1e-10, nrm[mask][:, 2])
    gy[mask] = -nrm[mask][:, 1] / np.maximum(1e-10, nrm[mask][:, 2])
    Fx, Fy, Fr = np.fft.fft2(fx, size), np.fft.fft2(fy, size), np.fft.fft2(fr, size)
    Gx = np.fft.fft2(gx, size)
    Gy = np.fft.fft2(gy, size)
    numerator = np.conj(Fx)*Gx + np.conj(Fy)*Gy
    denominator = np.square(np.abs(Fx))+np.square(np.abs(Fy))+lmda*np.square(np.abs(Fr))
    Fz = numerator / np.maximum(1e-10, denominator)
    Fz[0, 0] = 0
    gz = np.real(np.fft.ifft2(Fz))
    return gz


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
