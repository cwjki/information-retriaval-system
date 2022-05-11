from typing import List


class Database:
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index) -> str:
        return self.documents[index]

    def __iter__(self) -> 'DatabaseIterator':
        return DatabaseIterator(self)


class DatabaseIterator:
    def __init__(self, database: Database) -> None:
        self.__index: int = -1
        self.__database: Database = database

    def __next__(self) -> str:
        self.__index += 1
        if self.__index == len(self.__database):
            raise StopIteration
        return self.__database[self.__index]


class Document:
    def __init__(self) -> None:
        self.title = ''
        self.text = ''
        self.author = ''

    def __repr__(self) -> str:
        return f'Title: {self.title} <br> Author: {self.author} <br> Text: {self.text} <br>'

    def __str__(self) -> str:
        return self.__repr__()
