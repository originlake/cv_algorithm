### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    n = np.zeros(imgs[0].shape)
    I = None
    for img in imgs:
        gray_img = np.sum(img, axis=2)[np.where(mask)]
        if I is None:
            I = np.array([gray_img])
        else:
            I = np.concatenate((I,[gray_img]))
    n_valid = np.transpose(np.linalg.solve(np.dot(np.transpose(L), L), np.dot(np.transpose(L), I)))
    n_valid = n_valid/ np.maximum(1e-10, np.sqrt(np.sum(n_valid**2, axis=1)).reshape(-1, 1))
    n[np.where(mask)] = n_valid

    return n

# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    # rho = sum(<l,n>*I)/sum(<l,n>^2)
    rho = np.zeros(imgs[0].shape)
    I = None
    mask = np.where(mask)
    n = nrm[mask]
    for img in imgs:
        img_val = img[mask]
        if I is None:
            I = np.array([img_val])
        else:
            I = np.concatenate((I,[img_val]))
    I = np.swapaxes(I, 0, 1)
    nI = np.dot(n, np.transpose(L))[:, :, np.newaxis]
    numerator = np.sum(I*nI, axis=1)
    denominator = np.maximum(1e-10, np.sum(nI**2, axis=1))
    rho[mask] = numerator/denominator
    return rho
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
