<<<<<<< HEAD
=======
import numpy as np
from PIL import Image

def load_image(file: str) -> np.ndarray[np.uint8]:
    img = Image.open(file)
    img.load()

    data = np.asarray(img, dtype=np.uint8)
    return data

def save_image(file: str, data: np.ndarray[np.uint8]):
    img = Image.fromarray(data.astype(np.uint8), mode="RGB")
    img.save(file)

def padded_shape(shape: tuple[int]) -> tuple[int]:
    res = [n + 4 - n % 4 for n in shape]
    res[2] = 3
    return tuple(res)
>>>>>>> main


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

<<<<<<< HEAD
    (h, w) = data.shape
    fixed_w = 4 - (w % 4)

    data = np.hstack((data, np.zeros((h, fixed_w))))
    data = np.vstack((data, np.zeros((4 - (h % 4), w + fixed_w))))
    return data
=======
    (h, w, _) = data.shape
    result = np.zeros(padded_shape(data.shape))
    result[0:h, 0:w, 0:3] = data
    return result
>>>>>>> main

def remove_padding(data: np.ndarray[np.uint8], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Retire le padding des octets qui représentent une image.
    """

<<<<<<< HEAD
    (h, w) = shape

=======
    (h, w, _) = shape
>>>>>>> main
    return data[0:h, 0:w]

def split(data: np.ndarray[np.uint8]) -> list[np.ndarray[np.uint8]]: 
    """
    Découpe les octets qui représentent une image en des blocs de forme 4x4
    """

<<<<<<< HEAD
    (h, w) = data.shape
=======
    (h, w, _) = data.shape
>>>>>>> main
    
    res = []
    for x in range(0, w, 4):
        for y in range(0, h, 4):
            res.append(data[y:y+4, x:x+4])

    return res

def join(blocks: list[np.ndarray[np.uint8]], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Reconstitue les octets qui représentent une image à partir de blocs de forme 4x4
    """

<<<<<<< HEAD
    [h, w] = map(lambda n: n + 4 - n % 4, shape)

    data = np.zeros((h, w))
=======
    (h, w, _) = padded_shape(shape)

    data = np.zeros((h, w, 3))
>>>>>>> main

    for x in range(0, w, 4):
        for y in range(0, h, 4):
            block = blocks[(x // 4) * (h // 4) + (y // 4)]
            data[y:y+4, x:x+4] = block

    return data

<<<<<<< HEAD
mat = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
], np.uint8)
shape = mat.shape

blocks = split(add_padding(mat))
print(blocks)

original = remove_padding(join(blocks, shape), shape)
print(original)





#couleurs a et b , des matrices:
def convert_color_to_565(color):
    # Conversion des valeurs RVB sur 8 bits en valeurs RVB sur 5:6:5 bits
    r = int(np.round(color[0] * 31 / 255))
    g = int(np.round(color[1] * 63 / 255))
    b = int(np.round(color[2] * 31 / 255))
    return np.array([r, g, b])

def build_palette(a, b):
    # Convertir les couleurs a et b en valeurs RVB sur 5:6:5 bits
    a_565 = convert_color_to_565(a)
    b_565 = convert_color_to_565(b)
    
    # Construire le tableau de 4 couleurs en utilisant les valeurs RVB sur 5:6:5 bits
    palette = [
        a_565,
        np.round((2 * a_565 + b_565) / 3).astype(int),
        np.round((a_565 + 2 * b_565) / 3).astype(int),
        b_565
    ]
    
    return palette

# Exemple d'utilisation
a = np.array([255, 0, 0])  # Couleur rouge
b = np.array([0, 0, 255])  # Couleur bleue
palette = build_palette(a, b)
print(palette)


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

#Ex5:
def prochepixel(p,q):
    np.linalg.norm(p.astype(int) - q)
#-----------SECTION TEST------------------------
=======
def truncate(n: int, p: int):
    """
    Tronque un entier en retirant les p bits les moins significatifs de l'entier
    """
    return n >> p

def create_palette(a: int, b: int):
    """
    Renvoie une palette à partir de deux pixels tronqués
    """
    return [np.round(p) for p in [a, 2 * a / 3 + b / 3, a / 3 + 2 * b / 3, b]]

def truncate_pixel(px: np.ndarray[np.uint8]) -> int:
    """
    Renvoie un pixel tronqué, un entier codé de la manière suivante : 5 bits pour le rouge, 6 bits pour le vert, 5 bits pour le bleu
    """
    (r, g, b) = px
    tr = truncate(r, 3)
    tg = truncate(g, 2)
    tb = truncate(b, 3)

    return tb | tg << 5 | tr << 11

def detruncate_pixel(px: int) -> np.ndarray[np.uint8]:
    """
    Renvoie un pixel assez proche de l'original à partir d'un entier de 16 bits
    """
    r = (px >> 11) << 3
    g = (px >> 5 & 0x3F) << 2
    b = (px & 0x1F) << 3

    return np.array([r, g, b], dtype=np.uint8)

def find_nearest(palette: np.ndarray[np.uint8], px: np.ndarray[np.uint8]):
    nearest = 0
    dist = np.inf
    for i in range(len(palette)):
        new_dist = np.linalg.norm(px.astype(int) - palette[i])
        print(new_dist)
        if new_dist < dist:
            dist = new_dist
            nearest = i

    return nearest

mat = load_image("image.jpg")
shape = mat.shape

blocks = split(add_padding(mat))
removed = remove_padding(join(blocks, shape), shape)

save_image("output.jpg", removed)
>>>>>>> main
