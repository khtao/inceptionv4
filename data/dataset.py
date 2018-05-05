from torch.utils.data import dataset
import os
from glob import glob
import pretrainedmodels.utils as utils
import numpy as np


class PathologyDataset(dataset.Dataset):
    """Face Landmarks dataset."""
    '''
    ./dataset/
        train/
            neg/
            pos/
        test/
            neg/
            pos/
    '''

    def __init__(self, root_dir, mode="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): test or train
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        neg_path = os.path.join(root_dir, mode, "neg")
        pos_path = os.path.join(root_dir, mode, "pos")
        self.neg_list = glob(os.path.join(neg_path, "*.tif"))
        self.pos_list = glob(os.path.join(pos_path, "*.tif"))
        self.transform = transform

    def __len__(self):
        return len(self.neg_list) + len(self.pos_list)

    def __getitem__(self, idx):
        if idx < len(self.neg_list):
            image_path = self.neg_list[idx]
            label = np.array(0).reshape([1, 1])
        else:
            image_path = self.pos_list[idx-len(self.neg_list)]
            label = np.array(1).reshape([1, 1])

        input_image = utils.LoadImage()(image_path)
        # hot_label = np.zeros(2, dtype=np.int32)
        # hot_label[label] = 1
        if self.transform:
            input_image = self.transform(input_image)

        return input_image, label.astype(np.long)
