import re
from typing import List
from .database import Document


class CranfieldParser:
    def __init__(self) -> None:
        pass

    def parse(self, file):
        with open(file, 'r') as f:
            data = ''.join(f.readlines())
        documents: List[str] = re.split('.I \d', data)
        documents = [document.strip() for document in documents]
        documents = [document for document in documents if not document == '']
        return [self.process_document(document) for document in documents]

    def process_document(self, document):
        lines: List[str] = document.split('\n')
        lines = list(map(lambda x: x.strip(), lines))
        separators: List[(int, str)] = []
        for separator in ['.T', '.A', '.B', '.W']:
            try:
                index = lines.index(separator)
                separators.append((index, separator))
            except ValueError:
                pass
        return self.create_document(lines, separators)

    def create_document(self, lines: List[str], separators):
        document: Document = Document()
        sections = sorted(separators)
        sections.append((len(lines), ''))
        for i, (index, separator) in enumerate(sections):
            if separator == '':
                break
            begin = index + 1
            end = sections[i+1][0]
            data = ''.join(lines[begin: end])
            if separator == '.T':
                document.title = data
            elif separator == '.W':
                document.text = data
            elif separator == '.A':
                document.author = data
        return document
