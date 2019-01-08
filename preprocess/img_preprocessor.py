import numpy as np
from PIL import Image


def preprocess_img(img):
    img = np.asarray(img).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def __conv_rgba2_rgb(image):
    image.load()  # required for png.split()
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    image = background.convert('L')

    return image


def preprocess_img_from_file(path_to_img):
    img = Image.open(path_to_img)
    temp = np.asarray(img).astype(np.float32)
    if len(temp.shape) == 3 and temp.shape[2] == 4:
        img = __conv_rgba2_rgb(img)
    elif len(temp.shape) == 2:
        pass
    else:
        raise Exception('Unexpected shape of image: %s' % (img.shape))
    return preprocess_img(img)
