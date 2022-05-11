from cmath import log
from src.inverted_index import InvertedIndex
from src.dataset import Dataset
from math import log10


class VectorSpaceModel:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.index = InvertedIndex(dataset)

    def compute_tf(self, document_index: int, term: str) -> float:
        term_frequency = self.index.get_term_frequency_document(
            document_index, term)
        max_frequency = self.index.get_max_term_frequency_document(
            document_index)
        return term_frequency / max_frequency

    def compute_idf(self, term: str) -> float:
        total_documents = self.index.get_total_documents()
        document_frequency = self.index.get_amount_document_with_term(term)
        return log10(total_documents, document_frequency)

    def compute_wight(self, document_index: int, term: str) -> float:
        tf = self.compute_tf(document_index, term)
        idf = self.compute_idf(term)
        return tf * idf

    
