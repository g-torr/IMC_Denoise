import os

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

markers_already_processed = [file.replace('weights_', '').replace('.hdf5', '') for file in os.listdir('trained_weights') if file.startswith('weights_') and file.endswith('.hdf5')]

# Iterate over marker_list and execute foo.py for each channel_name
for channel_name in marker_list:
    if channel_name not in markers_already_processed:
        # Assuming foo.py is in the same directory as this script
        command = f"python IMC_Denoise_Train.py --channel_name {channel_name} --Raw_directory {Raw_directory}"
        os.system(command)

