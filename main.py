import numpy as np
import math
from PIL import Image

WELCOME = """
______      _   _                  ______  _____  __    _____           _ 
| ___ \    | | | |                 | ___ \/  __ \/  |  |_   _|         | |
| |_/ /   _| |_| |__   ___  _ __   | |_/ /| /  \/`| |    | | ___   ___ | |
|  __/ | | | __| '_ \ / _ \| '_ \  | ___ \| |     | |    | |/ _ \ / _ \| |
| |  | |_| | |_| | | | (_) | | | | | |_/ /| \__/\_| |_   | | (_) | (_) | |
\_|   \__, |\__|_| |_|\___/|_| |_| \____/  \____/\___/   \_/\___/ \___/|_|
       __/ |                                                              
      |___/                                                               
"""

def load_image(file: str) -> np.ndarray[np.uint8]:
    """
    Charge une image depuis le disque et renvoie les données dans un tableau numpy
    """
    img = Image.open(file)
    img.load()

    data = np.asarray(img, dtype=np.uint8)
    return data

def save_image(file: str, data: np.ndarray[np.uint8]):
    """
    Enregistre une image sur le disque depuis les données d'un tableau numpy
    """
    img = Image.fromarray(data.astype(np.uint8), mode="RGB")
    img.save(file)

def padded_shape(shape: tuple[int]) -> tuple[int]:
    """
    Renvoie les dimensions d'une image avec la correction de remplissage
    """
    res = [math.ceil(n / 4) * 4 for n in shape]
    res[2] = 3
    return tuple(res)

def add_padding(data: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """
    Ajoute du remplissage sur les octets qui représentent une image, afin que ses dimensions soient multiples de 4
    """

    (h, w, _) = data.shape
    result = np.zeros(padded_shape(data.shape), dtype=np.uint8)
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

def truncate(n: int, p: int):
    """
    Tronque un entier en retirant les p bits les moins significatifs de l'entier
    """
    return n >> p

def create_palette(a: int, b: int):
    """
    Renvoie une palette à partir de deux pixels tronqués
    """
    return [np.round(p).astype(int) for p in [a, 2 * a / 3 + b / 3, a / 3 + 2 * b / 3, b]]

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
    """
    Renvoie l'index de la palette le plus proche du pixel
    """
    nearest = 0
    dist = np.inf
    for i in range(len(palette)):
        new_dist = np.linalg.norm(px.astype(int) - detruncate_pixel(palette[i]))
        if new_dist < dist:
            dist = new_dist
            nearest = i

    return nearest

def encode_block(block: np.ndarray[np.uint8], palette: np.ndarray[np.uint8], a: int, b: int) -> int:
    """
    Crée un patch à partir d'un block, une palette, et deux couleurs a et b.
    """
    res = 0
    for x in range(0, 4):
        for y in range(0, 4):
            idx = find_nearest(palette, block[x, y])
            res |= idx << ((x * 4 + y) * 2)

    res |= int(b) << 32
    res |= int(a) << 48

    return res

def decode_block(patch: int) -> np.ndarray[np.uint8]:
    """
    Crée un bloc à partir d'un patch.
    """
    a = patch >> 48
    b = patch >> 32 & 0xFFFF

    palette = create_palette(a, b)

    data = np.zeros((4, 4, 3))

    for x in range(4):
        for y in range(4):
            i = x * 4 + y
            idx = (patch >> i * 2) & 0b11

            data[y, x] = detruncate_pixel(palette[idx])
    
    return data

def ab_minmax(block: np.ndarray[np.uint8]) -> tuple[int, int]:
    """
    Détermine les couleurs a et b depuis un bloc
    """
    m = [255, 255, 255]
    M = [0, 0, 0]

    for i in range(4):
        for j in range(4):
            color = block[i, j]

            for k in range(3):
                if color[k] < m[k]:
                    m[k] = color[k]
                
                if color[k] > M[k]:
                    M[k] = color[k]
    
    return truncate_pixel(m), truncate_pixel(M)

def encode(path: str, patches: list[int], shape: tuple[int, int]):
    """
    Encode une image au format BC1 et la stocke dans un fichier
    """

    (h, w, _) = shape

    with open(path, "w") as f:
        # Headers
        f.write("BC1\n")
        f.write(f"{h} {w}\n")

        # Data
        for patch in patches:
            f.write(f"{str(patch)}\n")

def decode(path: str) -> tuple[np.ndarray[np.uint8], tuple[int, int]]:
    """
    Décode le fichier au chemin d'accès spécifié
    """

    with open(path, "r") as f:
        assert f.readline() == "BC1\n"
        
        [h, w] = [int(n) for n in f.readline().split()]
        patches = [int(line) for line in f.readlines()]

        return join([decode_block(patch) for patch in patches], (h, w, 3)), (h, w, 3)


def main():
    print(WELCOME)

    print("What would you like to do?")
    print("* encode")
    print("* decode")

    cmd = input("> ")

    print("Enter file path")
    path = input("> ")

    match cmd.lower():
        case "encode":
            mat = load_image(path)
            shape = mat.shape
            blocks = split(add_padding(mat))

            patches = []
            for block in blocks:
                a, b = ab_minmax(block)
                palette = create_palette(a, b)

                patches.append(encode_block(block, palette, a, b))

            encode("compressed.bc1", patches, shape)
        case "decode":
            data, shape = decode("compressed.bc1")
            save_image("compressed.jpg", remove_padding(data, shape))

    print("Done")

if __name__ == '__main__':
    main()