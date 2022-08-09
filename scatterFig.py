import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from PIL import Image

_PALETTE = [((0.00, 0.25), (15, 31, 151)),  # darkblue
            ((0.25, 0.50), (2, 77, 178)),  # mediumblue
            ((0.50, 1.00), (58, 108, 255)),  # lightblue
            ((1.00, 2.00), (5, 182, 5)),  # green
            ((2.00, 4.00), (220, 205, 7)),  # yellow
            ((4.00, 8.00), (255, 154, 0)),  # orange
            ((8.00, 16.00), (202, 49, 51)),  # fuchsia
            ((16.00, 32.00), (255, 2, 255)),  # white
            ((-2.00, -0.5), (150, 150, 150))]  # lightgrey


def get_palette():
    palette = _PALETTE
    img_palette = [0, 0, 0] + [item for sublist in palette for item in sublist[1]]
    return img_palette


def figsave(df, label1=[], label2=[], v=0, p=10, name='-dual.png'):
    print('-----------------------', str(label1) +
          '.png', '--------------------------')
    plt.subplots(figsize=(17, 8.5))
    plt.scatter(df.x, df.y, s=1, c=df[label1], cmap='Oranges', vmin=0, vmax=0.5 * p)
    plt.grid()
    # valididx = np.where(df[label2] > 0)
    valididx = np.where(df[label1] - df[label2] > v)
    print(valididx[0], df[label2][valididx[0]] - df[label1][valididx[0]])
    gt = plt.scatter(df.x[valididx[0]], df.y[valididx[0]], s=30,
                     c=df[label1][valididx[0]] - df[label2][valididx[0]], cmap='Greens', vmin=0, vmax=0.5 * p)
    valididx = np.where(df[label1] - df[label2] < -1 * v)
    print(valididx[0], df[label1][valididx[0]] - df[label2][valididx[0]])
    lt = plt.scatter(df.x[valididx[0]], df.y[valididx[0]], s=30,
                     c=df[label2][valididx[0]] - df[label1][valididx[0]], cmap='Oranges', vmin=0, vmax=0.5 * p)
    hc = plt.colorbar(gt)
    hc.set_label('lower')
    lc = plt.colorbar(lt)
    lc.set_label('higher')
    # plt.clim(-5.0, 5.0)
    plt.title(str(label2) + '-d-' + str(v))
    plt.savefig(name)


def display(df, label1=[], i=0, v=0, name='-p.png'):
    plt.subplots(figsize=(17, 8.5))
    gt = plt.scatter(df.x, df.y, s=250, c=df[label1], cmap='rainbow')
    plt.grid()
    # plt.clim(-1.0, 1.0)
    hc = plt.colorbar(gt)
    hc.set_label('higher')
    plt.title(str(label1) + '-d-' + str(v))
    plt.savefig(name + str(label1) + '.png')


def data_to_img(data, img_filename, cols, rows):
    data = np.array(data)
    im = Image.fromarray(np.uint8(np.array(data).reshape(cols, rows)), 'P')
    png_palette = get_palette()
    im.putpalette(png_palette)
    im.save(str(img_filename) + '.png', transparency=0)
