from typing import List



class Dataset:
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index) -> str:
        return self.documents[index]

    def __iter__(self) -> 'DatasetIterator':
        return DatasetIterator(self)


class DatasetIterator:
    def __init__(self, dataset: Dataset) -> None:
        self.index: int = -1
        self.dataset: Dataset = dataset

    def __next__(self) -> str:
        self.index += 1
        if self.index == len(self.dataset):
            raise StopIteration
        return self.dataset[self.index]


class Document:
    def __init__(self) -> None:
        self.title = ''
        self.text = ''
        self.author = ''

    def __repr__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.__repr__()
