# Multimodal data loader that streams data from disk. Can load, process and return image data

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import List
import utils as utils
import math
import random

class CustomImageDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, file_names_csv: str, images_directory: str, batch_size: int = 1, randomize: bool = False):
        self.file_names_csv = file_names_csv
        self.images_directory = images_directory
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = self.get_length() // self.batch_size  # Updated initialization

    def get_length(self):
        file_names_df = pd.read_csv(os.path.join('data', self.file_names_csv))
        self.num_batches_per_epoch = math.ceil(len(file_names_df) / self.batch_size)
        return len(file_names_df)
    
    def randomize_dataset(self):
        """
        This function randomizes the dataset
        """
        file_names_df = pd.read_csv(self.file_names_csv)
        self.file_names = file_names_df['filename'].tolist()
        random.shuffle(self.file_names)

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
        if self.randomize:
            self.randomize_dataset()

        # Load file names from the CSV
        csv_path = os.path.join('data', self.file_names_csv)
        file_names_df = pd.read_csv(csv_path)
        file_names = file_names_df['filename'].tolist()

        # Create full file paths using images_directory
        file_paths = [os.path.join(self.images_directory, filename) for filename in file_names]

        # split dataset into a sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batch_files = file_paths[b_idx * self.batch_size : (b_idx+1) * self.batch_size]
            images = [Image.open(file_name) for file_name in batch_files]
            # Perform any necessary image transformations here
            # Resize images to 128x128, convert to numpy array, and normalize
            images = [np.array(image.resize((128, 128))) / 255.0 for image in images]
            images = np.array(images)
            images = torch.tensor(images)
            
            batches.append({
                'images': images,
                'batch_idx': b_idx,
            })
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter is None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()

        return batch


class CustomDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()

        return batch





