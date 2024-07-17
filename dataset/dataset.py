# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.utils import min_version, optional_import

import scipy.io

cp, _ = optional_import("cupy")
lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")
kvikio_numpy, _ = optional_import("kvikio.numpy")

# Helper function to apply a transform
def apply_transform(transform, data):
    return transform(data)

class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of .mat file paths::

        ['file1.mat', 'file2.mat', 'file3.mat']
    """

    def __init__(self, data: Sequence, transform: Callable | None = None) -> None:
        """
        Args:
            data: List of .mat file paths to load and transform to generate dataset for model.
            transform: A callable data transform on input data.
        """
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)
    
    def _load_image(self, file_path: str):
        """
        Load the 'image' data from the given .mat file.
        """
        mat_data = scipy.io.loadmat(file_path)
        image_data = mat_data.get('image')
        if image_data is None:
            raise ValueError(f"'image' key not found in {file_path}")
        return image_data

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        file_path = self.data[index]
        image_data = self._load_image(file_path)
        return apply_transform(self.transform, image_data) if self.transform is not None else image_data


    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)