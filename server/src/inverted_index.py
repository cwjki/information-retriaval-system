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
        return self.index[term][document_index]

    def get_max_term_frequency_document(self, document_index: int) -> int:
        return self.max_term_frequency_document[document_index]

    def get_amount_document_with_term(self, term: str) -> int:
        return len(self.index[term])

    def get_total_documents(self) -> int:
        return len(self.dataset)


class DocumentTerm:
    def __init__(self, document_index: int) -> None:
        self.document_index: int = document_index
        self.term_frequency: int = 1

    @property
    def get_document_index(self) -> int:
        return self.document_index

    @property
    def get_term_frequency(self) -> int:
        return self.term_frequency
