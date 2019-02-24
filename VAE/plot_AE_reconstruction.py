import keras
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 256


encoder = keras.models.load_model('encoder')
generator = keras.models.load_model('decoder')

n = 10
img_rows, img_cols, img_chns = 180, 180, 1
figure = np.zeros((img_rows * n, img_cols * n))

for i in range(n):
    for j in range(n):
        z_sample = np.random.normal(size=latent_dim).reshape(1, latent_dim)
        x_decoded = generator.predict(z_sample, batch_size=1)
        digit = x_decoded.reshape(img_rows, img_cols)

        d_x = i * img_rows
        d_y = j * img_cols
        figure[d_x:d_x + img_rows, d_y:d_y + img_cols] = digit[:, :]

plt.figure(figsize=(12, 16))
plt.grid(b=False)
plt.imshow(figure)
plt.gray()
plt.savefig('AE synthetic images.png')
plt.show()