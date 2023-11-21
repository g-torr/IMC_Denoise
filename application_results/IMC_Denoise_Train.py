#!/usr/bin/env python
# coding: utf-8

# # IMC-Denoise: a content aware denoising pipeline to enhance imaging mass cytometry

# Here we will show an example for denoising the images with marker CD14 from our own human bone marrow IMC dataset. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator, load_training_patches
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
import os
import argparse

# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





def generate_patches(channel_name,Raw_directory,Save_directory,n_neighbours,n_iter,window_size ):
    '''Generate images patches and save them to disk'''

    # ### Training data preparation
    # Next, we use our raw images to build a training set.
    # Note: 
    # 1. The channel name must be consistant with the corresponding channel name in the image file names. For example, in our dataset, CD14 is conjucted with 144Nd. If the images with marker CD14 need to be denoised, the channel name will be set as its isotope name "144Nd".
    # 2. Raw_directory is the folder of all the raw images used for generating training set. Its subfolders are the imagesets of different tissues. The subfolders contains the images from all the channels of the same tissue. 
    # <b><br>Data_structure example:
    # <b><br>|---Raw_image_directory
    # <br>|---|---Tissue1
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # <br>|---|---Tissue2
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # <br>             ...
    # <br>|---|---Tissue_m
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # </b>
    # 3. Save_directory is the folder used for saving generated training data. If None, it will be saved in the default folder. For CD14, the saved training set is "training_set_144Nd.npz".
    # 4. n_neighbour and n_lambda are the parameters from DIMR algorithm for hot pixel removal in the training set generation process. 4 and 5 are their defaults. If the defaults are changed, the corresponding parameter should be declared in DeepSNiF_DataGenerator(). Otherwise, they can be omitted.
    # 5. The DeepSNiF_DataGenerator class search all the CD14 images in raw image directory, split them into multiple 64x64 patches, and then augment the generated data. Note the very sparse patches are removed in this process.
    # 6. Here we will save the generated training set and later reload it.

    # Release memory
    if 'generated_patches' in globals():
        del generated_patches
    Raw_directory = "/home/giuseppe/devices/Delta_Tissue/IMC/split_channels_nohpf/" # change this directory to your Raw_image_directory.
     
    DataGenerator = DeepSNiF_DataGenerator(channel_name = channel_name, n_neighbours = n_neighbours, n_iter = n_iter,window_size = window_size )
    generated_patches = DataGenerator.generate_patches_from_directory(load_directory = Raw_directory)
    if DataGenerator.save_patches(generated_patches, save_directory = Save_directory):
        print('Data generated successfully!')










def load_and_train(channel_name,deepsnif, Save_directory):
    '''
    Load the generated training data from directory
    '''
    saved_training_set = 'training_set_'+channel_name+'.npz'
    train_data = load_training_patches(filename = saved_training_set, save_directory = Save_directory)
    print('The shape of the loaded training set is ' + str(train_data.shape))
    train_loss, val_loss = deepsnif.train(train_data)
    return train_loss, val_loss
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--channel_name", help = "channel used to generate training set, e.g. CD45", type = str)
    parser.add_argument("--Raw_directory", help = "The directory which contained raw IMC images used to generate training set", type = str)
    args = parser.parse_args()
    channel_name = args.channel_name
    Raw_directory = args.Raw_directory
    n_neighbours = 10 # Larger n enables removing more consecutive hot pixels. 
    n_iter = 3 # Iteration number for DIMR
    window_size = 5
    Save_directory = None # If None, it will be saved in the default folder.
    
    # ### DeepSNiF configuration and training
    # Define parameters for DeepSNiF training. If is_load_weights is True, the pre-trained model and pre-calculated range of the marker channel will be loaded. The range value is used for normalization in prediction.
    train_epoches = 5 # training epoches, which should be about 200 for a good training result. The default is 200.
    train_initial_lr = 1e-3 # inital learning rate. The default is 1e-3.
    train_batch_size = 128 # training batch size. For a GPU with smaller memory, it can be tuned smaller. The default is 128.
    pixel_mask_percent = 0.2 # percentage of the masked pixels in each patch. The default is 0.2.
    val_set_percent = 0.15 # percentage of validation set. The default is 0.15.
    loss_function = "I_divergence" # loss function used. The default is "I_divergence".
    weights_name = "weights_"+channel_name+".hdf5" # trained network weights name. If None, the weights will not be saved.
    loss_name = None # training and validation losses name, either .mat or .npz format. If not defined, the losses will not be saved.
    weights_save_directory = None # location where 'weights_name' and 'loss_name' saved.
    # If the value is None, the files will be saved in a sub-directory named "trained_weights" of the current file folder.
    is_load_weights = False # Use the trained model directly. Will not read from any saved ones.
    lambda_HF = 3e-6 # HF regularization parameter.
    #End configuration, do stuff
    
    generate_patches(channel_name,Raw_directory=Raw_directory,Save_directory=Save_directory,n_neighbours=n_neighbours,n_iter = n_iter ,window_size = window_size )
    deepsnif = DeepSNiF(train_epoches = train_epoches, 
                    train_learning_rate = train_initial_lr,
                    train_batch_size = train_batch_size,
                    mask_perc_pix = pixel_mask_percent,
                    val_perc = val_set_percent,
                    loss_func = loss_function,
                    weights_name = weights_name,
                    loss_name = loss_name,
                    weights_dir = weights_save_directory, 
                    is_load_weights = is_load_weights,
                    lambda_HF = lambda_HF)

    train_loss, val_loss = load_and_train(channel_name,deepsnif = deepsnif, Save_directory =  Save_directory)
    np.save('./trained_weights/loss_tv_'+channel_name+'npy',[train_loss, val_loss])
    del deepsnif
    tf.keras.backend.clear_session() #to free up memory
if __name__ == "__main__":
    main()