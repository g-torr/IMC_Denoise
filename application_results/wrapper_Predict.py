# %% [markdown]
# # IMC-Denoise: a content aware denoising pipeline to enhance imaging mass cytometry

# %% [markdown]
# Here we will show an example for denoising the images with marker CD14 from our own human bone marrow IMC dataset. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tp
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
import os
import argparse
# %%

def main():
    Raw_directory = "/home/giuseppe/devices/Delta_Tissue/IMC/split_channels_nohpf/"

    # Load one acquisition directory
    acquisition_dirs = [d for d in os.listdir(Raw_directory) if os.path.isdir(os.path.join(Raw_directory, d))]

    if acquisition_dirs:
        # Take the first directory in the list
        first_acq_dir = os.path.join(Raw_directory, acquisition_dirs[0])
        
        # Get a list of markers by removing the file extension from files in the first directory
        marker_list = [os.path.splitext(file)[0] for file in os.listdir(first_acq_dir) if file.endswith('.tiff')]
    else:
        raise ValueError('No file found in '+Raw_directory)
    save_directory = '/home/giuseppe/devices/Delta_Tissue/IMC/Img_Denoised'
    isExist = os.path.exists(save_directory)
    if not isExist:
        os.makedirs(save_directory)
    n_neighbours = 10 # Larger n enables removing more consecutive hot pixels. 
    n_iter = 3 # Iteration number for DIMR
    window_size = 5
    #marker_list = ['CD4','CD44' ,'CD163','CD16']
    for channel_name in marker_list:
        # ### Load the pre-trained denoisng model for a marker
        weights_name = "weights_"+channel_name+".hdf5" # trained network weights name. 
        weights_save_directory = None # location where 'weights_name' will be loaded. 
        command = f"python ../scripts/Predict_IMC_Denoise_batch.py --channel_name {channel_name} --load_directory {Raw_directory} --save_directory {save_directory} --weights_name weights_{channel_name}.hdf5 --slide_window_size {window_size} --network_size small --batch_size 1"
        os.system(command)
if __name__ == "__main__":
    main()
