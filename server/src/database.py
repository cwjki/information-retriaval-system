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
        self.__index: int
        self.__database: Database = database

    def __next__(self) -> str:
        if self.__index == len(self.__database):
            raise StopIteration
        self.__index += 1
        return self.__database[self.__index]


class Document:
    def __init__(self, title: str, text: str, author: str) -> None:
        self.title = title
        self.text = text
        self.author = author
