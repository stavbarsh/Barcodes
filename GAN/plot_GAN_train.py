import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

name = 'GAN Results.csv'
history = pd.read_csv(name)
x = history['epoch']




fig = plt.figure(figsize=(15, 15))
# summarize history for accuracy
ax = fig.add_subplot(3, 2, 1)
ax.plot(x, history['D acc real'][:], color='C0')
ax.set_title('Discriminator accuracy for real images')
# ax.set(xlabel='epoch')
ax = fig.add_subplot(3, 2, 3)
ax.plot(x, history['D acc gen'][:], color='C1')
ax.set_title('Discriminator accuracy for generated images')
# ax.set(xlabel='epoch')
ax = fig.add_subplot(3, 2, 5)
ax.plot(x, history['G acc'][:], color='C2')
ax.set_title('Generator accuracy')
ax.set(xlabel='epoch')
ax = fig.add_subplot(3, 2, 2)
ax.plot(x, history['D loss real'][:], color='C0')
ax.set_title('Discriminator loss for real images')
# ax.set(xlabel='epoch')
ax = fig.add_subplot(3, 2, 4)
ax.plot(x, history['D loss gen'][:], color='C1')
ax.set_title('Discriminator loss for generated images')
# ax.set(xlabel='epoch')
ax = fig.add_subplot(3, 2, 6)
ax.plot(x, history['G loss'][:], color='C2')
ax.set_title('Generator loss')
ax.set(xlabel='epoch')

plt.show()
fig.savefig("GAN Loss and Accuracy Plot")


