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

def add_padding(data: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """
    Ajoute du padding sur les octets qui représentent une image, afin que ses dimensions soient multiples de 4
    """

    (h, w, _) = data.shape
    result = np.zeros(padded_shape(data.shape))
    result[0:h, 0:w, 0:3] = data
    return result

def remove_padding(data: np.ndarray[np.uint8], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Retire le padding des octets qui représentent une image.
    """

    (h, w, _) = shape
    return data[0:h, 0:w]

def split(data: np.ndarray[np.uint8]) -> list[np.ndarray[np.uint8]]: 
    """
    Découpe les octets qui représentent une image en des blocs de forme 4x4
    """

    (h, w, _) = data.shape
    
    res = []
    for x in range(0, w, 4):
        for y in range(0, h, 4):
            res.append(data[y:y+4, x:x+4])

    return res

def join(blocks: list[np.ndarray[np.uint8]], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Reconstitue les octets qui représentent une image à partir de blocs de forme 4x4
    """

    (h, w, _) = padded_shape(shape)

    data = np.zeros((h, w, 3))

    for x in range(0, w, 4):
        for y in range(0, h, 4):
            block = blocks[(x // 4) * (h // 4) + (y // 4)]
            data[y:y+4, x:x+4] = block

    return data

mat = load_image("image.jpg")
shape = mat.shape

blocks = split(add_padding(mat))
removed = remove_padding(join(blocks, shape), shape)

save_image("output.jpg", removed)