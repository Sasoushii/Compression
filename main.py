import numpy as np

def add_padding(data: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """
    Ajoute du padding sur les octets qui représentent une image, afin que ses dimensions soient multiples de 4
    """

    (h, w) = data.shape
    fixed_w = 4 - (w % 4)

    data = np.hstack((data, np.zeros((h, fixed_w))))
    data = np.vstack((data, np.zeros((4 - (h % 4), w + fixed_w))))
    return data

def remove_padding(data: np.ndarray[np.uint8], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Retire le padding des octets qui représentent une image.
    """

    (h, w) = shape

    return data[0:h, 0:w]

def split(data: np.ndarray[np.uint8]) -> list[np.ndarray[np.uint8]]: 
    """
    Découpe les octets qui représentent une image en des blocs de forme 4x4
    """

    (h, w) = data.shape
    
    res = []
    for x in range(0, w, 4):
        for y in range(0, h, 4):
            res.append(data[y:y+4, x:x+4])

    return res

def join(blocks: list[np.ndarray[np.uint8]], shape: tuple[int, int]) -> np.ndarray[np.uint8]:
    """
    Reconstitue les octets qui représentent une image à partir de blocs de forme 4x4
    """

    [h, w] = map(lambda n: n + 4 - n % 4, shape)

    data = np.zeros((h, w))

    for x in range(0, w, 4):
        for y in range(0, h, 4):
            block = blocks[(x // 4) * (h // 4) + (y // 4)]
            data[y:y+4, x:x+4] = block

    return data

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