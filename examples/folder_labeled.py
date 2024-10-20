from dataset import Dataset
import os
from PIL import Image
import numpy as np

class FolderLabeled(Dataset):
    def __init__(self, root_dir: str = "") -> None:
        self.paths = FolderLabeled._get_files(root_dir)
        self.labels = list(set([label for _, label in self.paths]))

    @staticmethod
    def _get_files(root_dir: str) -> list[str]:
        paths = []
        labels = os.listdir(root_dir)
        for label in labels:
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                paths.append((os.path.join(label_dir, file), label))
        return paths

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int):
        path, label = self.paths[index]
        return np.array(Image.open(path)) / 255., self.labels.index(label)

if __name__ == "__main__":
    dataset = FolderLabeled(root_dir=r"assignment2-task5\kaggle_train_128\train_128")
    print(len(dataset))
    print(dataset[0])