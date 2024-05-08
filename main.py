

import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    toLoad= Image.open(filename)
    return np.array(toLoad)


def psnr(original, compressed):
    mse = np.mean((original.astype(int) - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
#---------START-------------

def add_padding(data: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """
    Ajoute du padding sur les octets qui représentent une image, afin que ses dimensions soient multiples de 4
    """

    (h, w) = data.shape

    data = np.hstack((data, np.zeros((h, 4 - (w % 4)))))
    data = np.vstack((data, np.zeros((4 - (h % 4), 4))))
    return data

def remove_padding(data: np.ndarray[np.uint8], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Retire le padding des octets qui représentent une image.
    """

    (h, w) = shape

    print(h, w)

    return data[0:h, 0:w]

mat = np.array([
    [0, 0, 1],
    [0, 1, 0],
], np.uint8)
shape = mat.shape

padded = add_padding(mat)
print(padded)

print(remove_padding(padded, shape))

#EXERCICE 4:
def tronque(n,p):
    #conversion n en bits
    i=0
    sum=""
    while n>0:
        sum+=str(n%2)
        n=n//2
        i+=1
    sum=sum[::-1]
    if p!=0:
        sum=sum[:-p]
    return(sum)

#-----------SECTION TEST------------------------
