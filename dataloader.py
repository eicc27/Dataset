from tqdm import tqdm
from dataset import Dataset
import numpy as np

class DataLoader:
    def __init__(self, dts: Dataset, batch_size: int, shuffle: bool = False) -> None:
        self.dts = dts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dts))
        self.idx = 0
        self.dims = self._get_dim()

    def _shuffle(self):
        np.random.shuffle(self.indexes)

    def _get_dim(self):
        sample = self.dts[0]
        if isinstance(sample, tuple):
            return len(sample)
        return 0
    
    def __len__(self):
        return len(self.dts) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == 0 and self.shuffle:
            self._shuffle()
            print("Shuffled")
        if self.idx + self.batch_size > len(self.dts):
            self.idx = 0
            raise StopIteration
        batch = [[] for self.dims in range(self._get_dim())]
        for _ in range(self.batch_size):
            data = self.dts[self.indexes[self.idx]]
            if self.dims == 0:
                batch.append(data)
            else:
                for i, d in enumerate(data):
                    batch[i].append(d)
        self.idx += self.batch_size
        if self.dims == 0:
            return np.array(batch)
        return tuple(np.array(b) for b in batch)
    
if __name__ == "__main__":
    from examples.folder_labeled import FolderLabeled
    dataset = FolderLabeled(root_dir=r"assignment2-task5\kaggle_train_128\train_128")
    # print(dataset[0])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    # print(loader.dims)
    for i, (X, y) in tqdm(enumerate(loader)):
        pass
    print("done")
    for i, (X, y) in tqdm(enumerate(loader)):
        pass