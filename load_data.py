"""Load image data and labels."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torchvision
from skimage import io
from torch.utils.data import Dataset


class CassavaLeafDataset(Dataset):
    """Dataloader class."""

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transform: torchvision.transforms = None,
        mode: str = "train",
    ) -> None:
        """Intialize required variables."""
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        """Return length of dataframe."""
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        """Get image and label."""
        image_name = self.df["image_id"][index]
        label = self.df["label"][index]
        image = io.imread(os.path.join(self.root_dir, image_name))
        image = np.moveaxis(image, -1, 0)

        # Apply transform if available.
        if self.transform:
            image = self.transform(image)
        return (image, label)
