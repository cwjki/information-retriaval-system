from typing import List, Set

from .vector_space_model import VectorSpaceModel


class Evaluator():
    def __init__(self, documents, cran_queries, cran_relevant_indexes, model: VectorSpaceModel) -> None:
        self.documents = documents
        self.cran_queries = cran_queries
        self.cran_relevant_indexes = cran_relevant_indexes
        self.model = model
        self.precissions = []
        self.recalls = []
        self.f1s = []

    def evaluate(self):
        self.compute_metrics()
        # print(f'PRECISSION -> {self.precissions}')
        # print(f'RECALL -> {self.recalls}')
        return self.average(self.precissions), self.average(self.recalls), self.average(self.f1s)

    def compute_metrics(self):
        for i, query in enumerate(self.cran_queries):
            query = self.cran_queries[i]
            cran_relevant_indexes = self.cran_relevant_indexes[i]

            # recuperar la misma cantidad de documetos que Cran
            # model_relevant_indexes = self.model.get_ranking_index(
            #     str(query), len(cran_relevant_indexes))

            # recuperar la cantidad prefijada de documentos
            model_relevant_indexes = self.model.get_ranking_index(str(query))

            # print(
            #     f'CRAN {i} indices relevantes -> {len(cran_relevant_indexes)}')
            # print(
            #     f'MODEL {i} indices relevantes -> {len(model_relevant_indexes)}')

            # print(f'CRAN relevant indexes -> {cran_relevant_indexes}')
            # print(f'MODEL relevant indexes -> {model_relevant_indexes}')

            self.precissions.append(self.compute_precission(
                cran_relevant_indexes, model_relevant_indexes))

            self.recalls.append(self.compute_recall(
                cran_relevant_indexes, model_relevant_indexes))

            self.f1s.append(self.compute_f1(
                cran_relevant_indexes, model_relevant_indexes))

    def compute_precission(self, cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        if len(cran_relevant_indexes) == 0:
            return 0

        cran_set = set(cran_relevant_indexes)
        model_set = set(model_relevant_indexes)
        rr = cran_set.intersection(model_set)
        ri = model_set.difference(cran_set)

        # print(f'CRAN RELEVANTES -> {cran_set}')
        # print(f'MODEL RELEVANTES -> {model_set}')
        # print(f'RR -> {rr}')
        # print(f'RI -> {ri}')

        return len(rr) / (len(rr) + len(ri))

    def compute_recall(self, cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        if len(cran_relevant_indexes) == 0:
            return 0

        cran_set = set(cran_relevant_indexes)
        model_set = set(model_relevant_indexes)
        rr = cran_set.intersection(model_set)
        nr = cran_set.difference(model_set)

        # print(f'RR -> {rr}')
        # print(f'NR -> {nr}')
        # print(f'NO RECUPERADOS RELEVANTES -> {len(nr)}')

        return len(rr) / (len(rr) + len(nr))

    def compute_f1(self, cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        precission = self.compute_precission(
            cran_relevant_indexes, model_relevant_indexes)
        recall = self.compute_recall(
            cran_relevant_indexes, model_relevant_indexes)

        if precission == recall == 0:
            return 0

        return 2 * (precission * recall) / (precission + recall)

    def average(self, values: List[float]) -> float:
        total = len(values)
        return sum(values) / total
