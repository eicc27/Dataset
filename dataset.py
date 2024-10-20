class Dataset:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        raise NotImplementedError