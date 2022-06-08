from typing import Dict, List
from src.dataset import Dataset
from src.preprocess_document import preprocces_document


class InvertedIndex:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.index: Dict[str, List[DocumentTerm]] = {}
        self.max_term_frequency_document: Dict[int, int] = [
            0 for _ in range(len(dataset))]

        for document_index, document in enumerate(dataset):
            self.analize_document(document_index, document.text)

    def analize_document(self, document_index: int, document: str):
        terms: List[str] = preprocces_document(document)
        max_frequency = 0
        for term in terms:
            frequency = self.update_frequency(document_index, term)
            max_frequency = max(max_frequency, frequency)
        self.max_term_frequency_document[document_index] = max_frequency

    def update_frequency(self, document_index: int, term: str) -> int:
        documents: List[DocumentTerm]
        try:
            documents = self.index[term]
            last_document = documents[-1]
            if last_document.document_index == document_index:
                last_document.term_frequency += 1
                return last_document.term_frequency
            else:
                documents.append(DocumentTerm(document_index))
                return 1
        except:
            documents = [DocumentTerm(document_index)]
            self.index[term] = documents
            return 1

    def get_term_frequency_document(self, document_index: int, term: str) -> int:
        document_term = self.find_document_term(term, document_index)
        return document_term.term_frequency

    def get_max_term_frequency_document(self, document_index: int) -> int:
        return self.max_term_frequency_document[document_index]

    def get_amount_document_with_term(self, term: str) -> int:
        try:
            return len(self.index[term])
        except KeyError:
            return 0.001

    def get_total_documents(self) -> int:
        return len(self.dataset)

    def find_document_term(self, term: str, document_index: int) -> 'DocumentTerm':
        documents = self.index[term]
        document_indexes = [d.document_index for d in documents]
        index = document_indexes.index(document_index)
        return documents[index]

    def remove_non_indexed_terms(self, terms: List[str]) -> List[str]:
        result = []
        for term in terms:
            try:
                _ = self.index[term]
                result.append(term)
            except KeyError:
                continue
        return result


class DocumentTerm:
    def __init__(self, document_index: int) -> None:
        self.document_index: int = document_index
        self.term_frequency: int = 1
