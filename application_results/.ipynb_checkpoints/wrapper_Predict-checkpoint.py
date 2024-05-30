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
import os, shutil
import logging
from tqdm import tqdm
import argparse
# %%
def aggregate_changes():
    def leap3_4():
        swap_from_leap4_to_leap3 =[14,15,16]
        swap_from_leap3_to_leap4 =[11,12,13]
        #old_name:new_name
        file_2_rename = {'Leap003_'+str(id):'Leap004_'+str(id) for id in swap_from_leap3_to_leap4}|{'Leap004_'+str(id):'Leap003_'+str(id) for id in swap_from_leap4_to_leap3}
        files_2_delete = []    
        return files_2_delete,file_2_rename
    
    def leap17_18():
        files_2_delete = ['Leap018_'+str(x)for x in range(14,25)] 
        return files_2_delete,{}
    def leap10_11():
        files_2_delete = ['Leap011_7']
        swap_from_leap11_to_leap10 =['6','8','9','10','11','12','13']
        swap_from_leap10_to_leap11 =['1','2','3','4','5']
        file_2_rename = {'Leap011_'+str(id):'Leap010_'+str(id) for id in swap_from_leap11_to_leap10}|{'Leap010_'+str(id):'Leap011_'+str(id) for id in swap_from_leap10_to_leap11}
        return files_2_delete,file_2_rename
    def leap24_25():
        files_2_delete = ['Leap024_'+str(x) for x in range(1,7)]+['Leap025_7']
        return files_2_delete,{}    
    def leap40():
        files_2_delete = ['Leap040_'+str(x) for x in [2,3,4,5,7]]
        return files_2_delete,{}

    functions = [leap3_4,leap17_18,leap10_11,leap24_25,leap40]
    files_2_rename = {}
    files_2_delete = []
    for f in functions:
        a,b = f()
        files_2_delete+=a
        files_2_rename|=b
    return files_2_delete,files_2_rename
files_2_delete,files_2_rename = aggregate_changes()

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
    marker_list.remove('Carboplatin') # we take care of Carboplatin on its own 
    save_directory = '/home/giuseppe/devices/Delta_Tissue/IMC/Img_Denoised/non_preprocessed'
    isExist = os.path.exists(save_directory)
    if not isExist:
        os.makedirs(save_directory)
    n_neighbours = 10 # Larger n enables removing more consecutive hot pixels. 
    n_iter = 3 # Iteration number for DIMR
    window_size = 5
    for channel_name in tqdm(marker_list):
        # ### Load the pre-trained denoisng model for a marker
        weights_name = "weights_"+channel_name+".hdf5" # trained network weights name. 
        weights_save_directory = None # location where 'weights_name' will be loaded. 
        command = f"python ../scripts/Predict_IMC_Denoise_batch.py --channel_name {channel_name} --load_directory {Raw_directory} --save_directory {save_directory} --weights_name weights_{channel_name}.hdf5 --slide_window_size {window_size} --network_size small --batch_size 5"
        os.system(command)
    ##rename the files according to Leor naming
    for file in files_2_delete:
        try:
            shutil.rmtree(save_directory+'/'+file)
        except FileNotFoundError:
            logging.warn(save_directory+'/'+file+' not found')
    for old,new in files_2_rename.items():
        try:
            os.rename(save_directory+'/'+old,save_directory+'/'+new)
        except:
            logging.warn(save_directory+'/'+old+' not found')
if __name__ == "__main__":
    main()
