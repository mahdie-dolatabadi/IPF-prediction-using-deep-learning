import torch # Torch modules for deep learning
import numpy as np # Numerical computations and array operations
import pandas as pd # Data manipulation and analysis
import cv2 # OpenCV for image processing
from skimage import exposure  # Module for image intensity adjustment and histogram-based operations, such as contrast enhancement and histogram equalization.
from PIL import Image  # Image handling
import pydicom  # Handling DICOM medical imaging files
import os # OS-level operations (file paths, directory handling)
import random # Random number generation
from configs import HyperParameters

params = HyperParameters("slope_train_vit_simple")
# Load a range of images from a CSV file into a Pandas DataFrame
images_range = pd.read_csv("images.csv")

# ==================== Set Random Seed for Reproducibility ====================

# Retrieve the seed value from the HyperP object
seed = params.seed  

# Set the seed for Python's built-in random module
random.seed(seed)  

# Set the PYTHONHASHSEED environment variable for hash-based operations 
# to ensure deterministic results
os.environ['PYTHONHASHSEED'] = str(seed)  

# Set the seed for NumPy's random number generator
np.random.seed(seed)  

# Set the seed for PyTorch to ensure reproducibility of results
torch.manual_seed(seed)  

# Function to load and preprocess a medical image from a DICOM file
def get_img(path):  
    # Read the DICOM file using pydicom
    d = pydicom.dcmread(path)
    
    # Extract the pixel array from the DICOM and enhance contrast using histogram equalization
    # Resize the processed image to a fixed resolution of 384x384 using OpenCV
    output = cv2.resize(exposure.equalize_hist(d.pixel_array), (384, 384))
    
    # Return the preprocessed image
    return output

# Function to load and preprocess a mask image
def get_mask(path):
    # Open the mask image using PIL (assumes it is a standard image format like PNG or JPEG)
    mask = Image.open(path)
    
    # Convert the mask to a NumPy array and resize it to 384x384
    mask = cv2.resize(np.array(mask), (384, 384))
    
    # Reshape the mask to ensure it has dimensions 384x384 (optional step for consistency)
    return mask.reshape(384, 384)

# Define a custom dataset class for training in PyTorch
class OSICData_train(torch.utils.data.Dataset):
    """
        Custom dataset class for loading and processing training data in the OSIC dataset.
        
        This dataset class filters out bad patient IDs, loads image data, corresponding masks, slopes, and tabular features, 
        and provides a random sampling of training data.

        Attributes:
        -----------
        BAD_ID : list
            A list of patient IDs to exclude from the dataset.
        keys : list
            A list of patient IDs to include in the dataset after filtering out bad IDs.
        a : dict
            A dictionary containing slopes for each patient.
        tab : dict
            A dictionary containing tabular features for each patient.
        all_data : list
            A list of image file paths for all selected patients.
        train_data : dict
            A dictionary where the key is a patient ID, and the value is a list of corresponding image file paths.
        mask_data : dict
            A dictionary where the key is a patient ID, and the value is a list of corresponding mask file paths.
        ten_percent : list
            A random sample (50%) of the data, used for training.

        Methods:
        --------
        __len__():
            Returns the number of samples (50% of available data) in the dataset.
        
        __getitem__(idx):
            Retrieves a specific sample (image, mask, tabular features, slope, and patient ID) at the given index `idx`.
        """
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, a, tab):
            
        """
        Initializes the OSICData_train dataset.

        Parameters:
        -----------
        keys : list
            A list of patient IDs to include in the dataset.
        a : dict
            A dictionary containing slopes for each patient.
        tab : dict
            A dictionary containing tabular features for each patient.
        """
        self.keys = [k for k in keys if k not in self.BAD_ID]
        
        # Store slopes (a) and tabular features (tab)
        self.a = a
        self.tab = tab
        
        # Initialize empty lists and dictionaries for storing data
        self.all_data = []
        self.train_data = {}
        self.mask_data = {}
        
        # Loop through each patient ID in the filtered list of keys
        for p in self.keys:  
            # Get the image range (min and max slices) for the current patient from the images_range DataFrame
            properties = images_range[images_range['ID'] == p]
            min_slice = properties['min']
            max_slice = properties['max'] 

            # List the image files and mask files for the patient, excluding the first and last 15% slices
            self.train_data[p] = os.listdir(f'{params.data_folder}/train/{p}/')[int(min_slice.iloc[0]):int(max_slice.iloc[0])] 
            self.mask_data[p] = os.listdir(f'{params.data_folder}/mask/{p}/')[int(min_slice.iloc[0]):int(max_slice.iloc[0])] 
            
            # Add each image file path to the all_data list
            for m in self.train_data[p]:
                self.all_data.append(p + '/' + m)
            
        # Sample 50% of the data randomly for training
        length = int(0.5 * len(self.all_data))
        self.ten_percent = random.sample(self.all_data, length)

    # Define the length of the dataset (number of samples)
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        --------
        int : The number of samples in the dataset (50% of the total available data).
        """
        return len(self.ten_percent)
    
    # Method to retrieve a specific sample from the dataset
    def __getitem__(self, idx):
        """
        Returns the number of samples in the dataset.

        Returns:
        --------
        int : The number of samples in the dataset (50% of the total available data).
        """
        # Initialize empty lists to hold masks, images, slopes, and tabular features
        masks = []
        x = []
        a, tab = [], [] 
        
        # Get the patient ID and image file path for the current sample
        k = self.ten_percent[idx]
        i = k[:25]  # Extract patient ID from the image file path
       
        try:
            # Construct the corresponding mask file path by replacing the extension with .jpg
            j = k[:-4] + '.jpg'
            
            # Load the image and mask using the get_img and get_mask functions
            img = get_img(f'{params.data_folder}/train/{k}')
            mask = get_mask(f'{params.data_folder}/mask/{j}')

            # Append the mask, image, slope, and tabular features to the respective lists
            masks.append(mask)
            x.append(img)
            a.append(self.a[i])
            tab.append(self.tab[i])
        except:
            # Print a message if there is an error loading the image or mask
            print(k, j)

        # Convert the masks, images, slopes, and tabular features into PyTorch tensors
        masks = torch.tensor(np.asanyarray(masks), dtype=torch.float32)
        x = torch.tensor(np.asanyarray(x), dtype=torch.float32)
        a = torch.tensor(np.asanyarray(a), dtype=torch.float32)
        tab = torch.tensor(np.asanyarray(tab), dtype=torch.float32)

        # Remove the extra dimension from the tabular feature tensor
        tab = torch.squeeze(tab, axis=0)

        # Return the mask, image, and tabular feature tensors, along with the slope and patient ID
        return [masks, x, tab], a, k 

class OSICData_test(torch.utils.data.Dataset):
    # List of patient IDs to exclude from the dataset
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    
    def __init__(self, keys, a, tab):
        # Filter out bad patient IDs from the dataset
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a  # Additional metadata for patients
        self.tab = tab  # Tabular data associated with patients

        self.train_data = {}  # Dictionary to store training image file names per patient
        self.mask_data = {}   # Dictionary to store corresponding mask file names per patient
        
        for p in self.keys:  # Iterate over patient IDs
            # Retrieve slice range information for the current patient
            properties = images_range[images_range['ID'] == p]
            min_slice = properties['min']
            max_slice = properties['max'] 

            # Get the total number of slices available for the patient
            p_n = len(os.listdir(f'{params.data_folder}/train/{p}/'))

            # Select a subset of slices by removing a percentage from the beginning and end
            self.train_data[p] = os.listdir(f'{params.data_folder}/train/{p}/')[int(params.strip_ct * p_n):-int(params.strip_ct * p_n)]
            self.mask_data[p] = os.listdir(f'{params.data_folder}/mask/{p}/')[int(params.strip_ct * p_n):-int(params.strip_ct * p_n)]

    def __len__(self):
        # Return the total number of patients in the dataset
        return len(self.keys)
    
    def __getitem__(self, idx):
        masks = []  # List to store mask images
        x = []  # List to store input images
        a, tab = [], []  # Lists to store corresponding metadata and tabular data
        
        k = self.keys[idx]  # Retrieve patient ID based on the provided index

        try:
            # Randomly select an image from the available slices for the patient
            i = np.random.choice(self.train_data[k], size=1)[0]
            j = i[:-4] + '.jpg'  # Convert filename to mask format (assuming a different extension)

            # Load the image and corresponding mask
            img = get_img(f'{params.data_folder}/train/{k}/{i}')
            mask = get_mask(f'{params.data_folder}/mask/{k}/{j}')

            # Append the data to respective lists
            masks.append(mask)
            x.append(img)
            a.append(self.a[k])
            tab.append(self.tab[k])
        except:
            print(k, i)  # Print the patient ID and image filename in case of an error

        # Convert lists to PyTorch tensors
        masks = torch.tensor(np.asanyarray(masks), dtype=torch.float32)
        x = torch.tensor(np.asanyarray(x), dtype=torch.float32)
        a = torch.tensor(np.asanyarray(a), dtype=torch.float32)
        tab = torch.tensor(np.asanyarray(tab), dtype=torch.float32)
        
        # Remove extra dimensions from tabular data
        tab = torch.squeeze(tab, axis=0)

        return [masks, x, tab], a, k  # Return the data along with the patient ID
