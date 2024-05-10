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










#IV ecriture dans un fichier
def imginfo(path, type_fichier, hauteur, largeur, codes_patchs):
    with open(path, "w") as f:
        f.write(type_fichier + "\n")
        dimensions = str(hauteur) + " " + str(largeur)
        f.write(dimensions + "\n")
        
        # Écrire les codes des patchs
        for code in codes_patchs:
            f.write(str(code) + "\n")

path="res.txt"
type_fichier = "BC1"
hauteur = 200
largeur = 300
codes_patchs=[1,3]

#V décompression

def lectureBC1(path):
    listeblocs = []
    with open(path, "r") as f:
        lignes = f.readlines()
        for ligne in lignes:
            listeblocs.append(int(ligne.strip()))
    return listeblocs

indices=[]
def transcouleurs(n):
    # Séparer l'entier en deux valeurs de couleur et un tableau d'indices
    # Prendre les 8 premiers bits, puis les 8 bits suivants, et les 8 derniers bits
    couleur1 = (n // (256 * 256)) % 256 # toutes les combinaisons possibles de deux valeurs sur 8 bits (de 0 à 255)
    couleur2 = (n // 256) % 256
    indice = n % 256
    # Mettre l'indice dans une liste
    indices.append(indice)
    # Renvoyer les valeurs séparées
    return couleur1, couleur2, indices

def reconstruire_image(blocs):
    image = []
    for bloc in blocs:
        couleur1, couleur2, indices = transcouleurs(bloc)
        for indice in indices: #ajouts des pixels 
            if indice == 0:
                image.append(couleur1)
            elif indice == 1:
                image.append(couleur2)
    return image








#------Zone TESTING-----


mat = load_image("image.jpg")
shape = mat.shape

blocks = split(add_padding(mat))
removed = remove_padding(join(blocks, shape), shape)

save_image("output.jpg", removed)

imginfo(path,type_fichier,hauteur,largeur,codes_patchs)

