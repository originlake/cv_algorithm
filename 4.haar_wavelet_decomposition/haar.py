import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    # Placeholder that does nothing
    kernal = 1/2*np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    if(img is not list):
        res = [img]
    img = res.pop()
    height, width = img.shape
    img = img.reshape(height // 2, 2, -1).swapaxes(1, 2).reshape(-1, 4)
    haarimg = np.transpose(np.dot(kernal, np.transpose(img))).reshape(height//2, width//2, 4)
    img = haarimg[:,:,0]
    H1 = haarimg[:,:,1]
    H2 = haarimg[:,:,2]
    H3 = haarimg[:,:,3]
    res=[[H1,H2,H3]]
    if nLev == 1:
        res.append(img)
        return res
    res.extend(im2wv(img, nLev-1))
    return res


def wv2im(pyr):
    # Placeholder that does nothing
    kernal = 1 / 2 * np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]])
    kernal = np.linalg.inv(kernal)
    pyr = pyr.copy()
    while len(pyr)>1:
        img = pyr.pop()
        [H1, H2, H3] = pyr.pop()
        h, w = img.shape
        haarimg = np.zeros([h, w, 4])
        for idx,ch in enumerate([img, H1, H2, H3]):
            haarimg[:,:,idx]=ch
        haarimg = haarimg.reshape(-1,4)
        img = np.transpose(np.dot(kernal, np.transpose(haarimg)))
        img = img.reshape(h, -1, 2).swapaxes(1,2).reshape(h*2, w*2)
        pyr.append(img)

    return pyr[-1]



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

# Zero out some levels and
for i in range(len(pyr)-1):
    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
