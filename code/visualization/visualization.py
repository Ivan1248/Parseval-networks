import numpy as np
import skimage, matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
import matplotlib as mpl


def get_color_palette(n, black_first=True):
    cmap = mpl.cm.get_cmap('hsv')
    colors = [np.zeros(3)] if black_first else []
    colors += [
        np.array(cmap(0.8 * i / (n - len(colors) - 1))[:3])
        for i in range(n - len(colors))
    ]
    return colors


def fuse_images(im1, im2, a):
    return a * im1 + (1 - a) * im2


def compose(images, format='0,0;1,0-1'):
    def get_image(frc):
        inds = [int(i) for i in frc.split('-')]
        assert (len(inds) <= 2)
        ims = [images[i] for i in inds]
        return ims[0] if len(ims) == 1 else fuse_images(ims[0], ims[1], 0.5)

    format = format.split(';')
    format = [f.split(',') for f in format]
    return np.concatenate([
        np.concatenate([get_image(frc) for frc in frow], 1) for frow in format
    ], 0)