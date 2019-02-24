# Multiple 1D Barcodes Detection and Decoding Using YOLO3, VAE and GAN
TAU Deep Leaning Class

---------------------------

## VAE and GAN
VAE based on https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-semi_supervised_learning/vae-m1-fit-mnist.ipynb

GAN based on https://github.com/eriklindernoren/Keras-GAN/blob/master/bgan/bgan.py

### Create invornment 

1. Download the files and install all the packages in the requirements.txt

2. Download the Muenster and synthetic h5 files and the trained weights from:
https://drive.google.com/drive/folders/1BmGZd2BLW9lyhak9G_2J_NP-MDQkHzmF?usp=sharing

### Run Models

Run barcodes_vae.py for VAE model and barcodes_bgan.py for GAN.

### Create Synthetic barcodes datasets

Run VAE\GAN Create Random Barcodes.ipynb. 

They both looks the same, just that the VAE version pack the data in splitted sets without the labels being one-hot encoded and the GAN version does.

---------------------------

## Yolo-v3

based on https://github.com/qqwweee/keras-yolo3

The datsets and weights are found in this link:
https://drive.google.com/open?id=1kNp--Asphf54hccRYhEPts8hcqSJOijt

### General

1. Upload all files to Drive, redefine work_DIR in each notebook you run.

### Detect barcodes in images:

1. Get the Trained weights file from the Drive directory (YOLO3_Barcode_Detection/qqwweee2/logs/003_1class/trained_weights_final.h5).

2. Define the parameters at the top of the notebook and RUN ALL in generate_with_cutting_option.ipynb 

### Train the network

1. If the original weights files are not present, RUN Main_notebook.ipynb to download them.

2. Define the parameters at the top of the notebook and RUN ALL in Train.ipynb 
