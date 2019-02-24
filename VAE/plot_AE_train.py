import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

name = 'YOLO TensorBoard_losses_graph.csv'

history = pd.read_csv(name)
x = history['epoch']

fig = plt.figure(figsize=(15, 15))
# summarize history for accuracy
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, history['training loss'][:])
ax.plot(x, history['validation loss'][:])
ax.set_title('Barcode YOLO3 Training')
ax.legend(['training loss', 'validation loss'], loc='upper right')
plt.xlabel('epoch')
plt.show()
fig.savefig("YOLO Plot")
# plt.close()

# name = 'AE.synthetic.log.csv'
# history = pd.read_csv(name)
# x = history['epoch']
#
# fig = plt.figure(figsize=(15, 15))
# # summarize history for accuracy
# ax = fig.add_subplot(3, 3, 1)
# ax.plot(x, history['loss'][:])
# ax.plot(x, history['val_loss'][:])
# ax.set_title('Barcode VAE Total Loss')
# ax.legend(['loss', 'val_loss'], loc='upper right')
# ax = fig.add_subplot(3, 3, 2)
# ax.plot(x, history['kl_loss'][:])
# ax.plot(x, history['val_kl_loss'][:])
# ax.set_title('Barcode VAE Cross-Entropy Loss')
# ax.legend(['kl_loss', 'val_kl_loss'], loc='upper right')
# ax = fig.add_subplot(3, 3, 4)
# ax.plot(x, history['logx_loss'][:])
# ax.plot(x, history['val_logx_loss'][:])
# ax.set_title('Barcode VAE KL Divergence Loss')
# ax.legend(['logx_loss', 'val_logx_loss'], loc='upper right')
#
# # plt.ylabel('perplexity')
# plt.xlabel('epoch')
# plt.show()
# fig.savefig("AE Plot")
# # plt.close()

