import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    outx, outy = X.shape[0], X.shape[1]
    # symmetric padding
    X = np.pad(X,((K,K),(K,K),(0,0)), 'constant',constant_values=0)
    W = np.zeros((outx, outy, 3))
    filteredX = np.zeros((outx, outy, 3))

    # method 1, loop over k
    for i in range(2*K+1):
        for j in range(2*K+1):
            # Space
            ws = ((K-i)**2+(K-j)**2)/(2 * sgm_s ** 2)
            # Intensity
            wi = abs(np.linalg.norm(X[i:i+outx, j:j+outy] - X[K:K+outx, K:K+outy], axis=2))**2 / (2*sgm_i**2)
            w = np.exp(-ws - wi)
            w[0:max(K-i, 0), :] = 0
            w[:,0:max(K-j,0)] = 0
            w[outx+K-i:outx,:] = 0
            w[:, outy+K-j:outy] = 0
            w = np.repeat(w[:,:,np.newaxis], 3, axis=2)
            W += w
            filteredX += w * X[i:i+outx, j:j+outy]
    # normalize
    filteredX /= W

    return filteredX


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9

print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))


print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,K,2,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
