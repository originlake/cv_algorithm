## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.
#
# Be careful about division by 0.
#
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    size = mask.shape
    mask = np.where(mask)
    fx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    fy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    fr = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 8 / 9, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])
    gx, gy = np.zeros(size), np.zeros(size)
    gx[mask] = -nrm[mask][:, 0] / np.maximum(1e-10, nrm[mask][:, 2])
    gy[mask] = -nrm[mask][:, 1] / np.maximum(1e-10, nrm[mask][:, 2])
    w = np.zeros(size)
    w[mask] = nrm[mask][:, 2]**2
    # init param
    Z = np.zeros(size)
    b = conv2(gx*w, -fx, 'same') + conv2(gy*w, -fy, 'same')
    r = b.copy()
    p = r.copy()
    for _ in range(500):
        Qp = conv2(conv2(p, fx, 'same') * w, -fx, 'same') + conv2(conv2(p, fy, 'same') * w, -fy, 'same') + \
             lmda * conv2(conv2(p, fr, 'same'), fr, 'same')
        alpha = np.sum(r**2)/np.sum(p*Qp)
        Z = Z + alpha*p
        r_new = r - alpha*Qp
        beta = np.sum(r_new**2)/np.sum(r**2)
        r = r_new
        p = r + beta * p
    return Z


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
Z = ntod(nrm,mask,1e-7)


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
