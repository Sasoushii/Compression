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