import json
import sys
import logging
import numpy as np
from pathlib import Path
from monai.config.type_definitions import PathLike
from typing import Callable, Sequence, Optional
from monai.data import CacheDataset
from monai.transforms import LoadImaged, Randomizable


class DicomDataset(Randomizable, CacheDataset):
    def __init__(
        self,
        dataset_root_dir: PathLike,
        section: str,
        transform: Sequence[Callable] | Callable = (),
        splits_json: str = "",
        seed: Optional[int] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache: bool = True,
    ) -> None:

        dataset_root_dir = Path(dataset_root_dir)
        if not dataset_root_dir.is_dir():
            raise ValueError("Root directory root_dir must be a directory.")
        logging.debug(f"📁 {dataset_root_dir = }")

        self.section = section

        if seed is not None:
            logging.warning(
                "Setting a fixed seed may lead to deterministic shuffling across "
                "training runs. Consider leaving seed as None for true randomness."
                " Set the seed only for debugging purpose."
            )
            self.set_random_state(seed=seed)

        self.indices: np.ndarray = np.array([])
        if splits_json == "":
            self.dataset_splits_json = "train_val_test_split.json"
        else:
            self.dataset_splits_json = splits_json
        self.json_path = dataset_root_dir / self.dataset_splits_json

        if not self.json_path.is_file():
            raise ValueError(
                f"The provided JSON file path {self.json_path} is invalid."
            )

        with open(self.json_path) as f:
            data = json.load(f)[self.section]
            length = len(data)
            self.c = np.arange(length)

        if transform == ():
            transform = LoadImaged(["image", "label"])

        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache,
        )

    def get_indicies(self) -> np.ndarray:
        return self.indices

    def randomize(self, data: np.ndarray) -> None:
        self.R.shuffle(data)
