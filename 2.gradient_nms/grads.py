import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    H = np.zeros(X.shape, dtype=np.float32)
    theta = np.zeros(X.shape, dtype=np.float32)
    sobelx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
    sobely = np.transpose(sobelx)

    Hx = conv2(X, sobelx, mode='same', boundary='symm')
    Hy = conv2(X, sobely, mode='same', boundary='symm')

    H = np.sqrt(Hx**2 + Hy**2)
    theta = np.arctan2(Hy, Hx)
    theta = 180 * theta / np.pi

    return H, theta

def nms(E,H,theta):
    #placeholder
    theta[theta < 0] += np.pi
    # get index of pixels need to be checked
    hori = np.where( np.logical_and(E, np.logical_or(theta < 22.5, theta >= 157.5)))        #(0,+1) (0,-1)    --
    diag1 =np.where( np.logical_and(E, np.logical_and(theta >= 22.5, theta < 67.5)))        #(-1,+1) (+1,-1)  /
    vert = np.where( np.logical_and(E, np.logical_and(theta >= 67.5, theta < 112.5)))       #(+1,0) (-1,0)    |
    diag2 = np.where( np.logical_and(E, np.logical_and(theta >= 112.5, theta < 157.5)))     #(-1,-1) (+1,+1)  \

    # exclude boundary pixels
    idx = np.where(np.logical_and(hori[1]>0, hori[1]<H.shape[1]-1))
    hori = (hori[0][idx], hori[1][idx])
    idx = np.where(np.logical_and(vert[0]>0, vert[0]<H.shape[0]-1))
    vert = (vert[0][idx], vert[1][idx])
    idx = np.where(np.logical_and(np.logical_and(diag1[0] > 0, diag1[1] > 0),
                                  np.logical_and(diag1[0] < H.shape[0]-1, diag1[1] < H.shape[1]-1)))
    diag1 = (diag1[0][idx], diag1[1][idx])
    idx = np.where(np.logical_and(np.logical_and(diag2[0] > 0, diag2[1] > 0),
                                  np.logical_and(diag2[0] < H.shape[0]-1, diag2[1] < H.shape[1]-1)))
    diag2 = (diag2[0][idx], diag2[1][idx])

    # set E
    E[hori] = np.logical_and(E[hori], np.logical_and(H[hori] > H[(hori[0] + 0, hori[1] + 1)],
                                                     H[hori] > H[(hori[0] + 0, hori[1] - 1)]))
    E[diag1] = np.logical_and(E[diag1], np.logical_and(H[diag1] > H[(diag1[0] - 1, diag1[1] + 1)],
                                                       H[diag1] > H[(diag1[0] + 1, diag1[1] - 1)]))
    E[vert] = np.logical_and(E[vert], np.logical_and(H[vert] > H[(vert[0] + 1, vert[1] + 0)],
                                                     H[vert] > H[(vert[0] - 1, vert[1] + 0)]))
    E[diag2] = np.logical_and(E[diag2], np.logical_and(H[diag2] > H[(diag2[0] + 1, diag2[1] + 1)],
                                                       H[diag2] > H[(diag2[0] - 1, diag2[1] - 1)]))
    return E

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.jpg'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
