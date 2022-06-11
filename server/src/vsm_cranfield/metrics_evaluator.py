from typing import List, Set

from .vector_space_model import VectorSpaceModel


class Evaluator():
    def __init__(self, documents, queries, relevant_indexes, model: VectorSpaceModel) -> None:
        self.documents = documents
        self.queries = queries
        self.relevant_indexes = relevant_indexes
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
        for i, query in enumerate(self.queries):
            query = self.queries[i]

            print(f'QUERY -> {query}')

            relevant_indexes = self.relevant_indexes[i]

            print(f'RELEVANT INDEXES -> {relevant_indexes}')

            # recuperar la misma cantidad de documetos que Cran
            # model_relevant_indexes = self.model.get_ranking_index(
            #     str(query), len(relevant_indexes))

            # recuperar la cantidad prefijada de documentos
            model_relevant_indexes = self.model.get_ranking_index(str(query))

            print(f'MODEL RELEVANT INDEXES -> {model_relevant_indexes}')

            # print(
            #     f'CRAN {i} indices relevantes -> {len(relevant_indexes)}')
            # print(
            #     f'MODEL {i} indices relevantes -> {len(model_relevant_indexes)}')

            # print(f'CRAN relevant indexes -> {relevant_indexes}')
            # print(f'MODEL relevant indexes -> {model_relevant_indexes}')

            self.precissions.append(self.compute_precission(
                relevant_indexes, model_relevant_indexes))

            self.recalls.append(self.compute_recall(
                relevant_indexes, model_relevant_indexes))

            self.f1s.append(self.compute_f1(
                relevant_indexes, model_relevant_indexes))

    def compute_precission(self, relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        if (len(relevant_indexes) == 0) or (len(model_relevant_indexes) == 0):
            return 0

        relevant_set = set(relevant_indexes)
        r_model_set = set(model_relevant_indexes)
        rr = relevant_set.intersection(r_model_set)
        ri = r_model_set.difference(relevant_set)

        # print(f'CRAN RELEVANTES -> {cran_set}')
        # print(f'MODEL RELEVANTES -> {model_set}')
        # print(f'RR -> {rr}')
        # print(f'RI -> {ri}')

        return len(rr) / (len(rr) + len(ri))

    def compute_recall(self, relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        if (len(relevant_indexes) == 0) or (len(model_relevant_indexes) == 0):
            return 0

        relevant_set = set(relevant_indexes)
        r_model_set = set(model_relevant_indexes)
        rr = relevant_set.intersection(r_model_set)
        nr = relevant_set.difference(r_model_set)

        # print(f'RR -> {rr}')
        # print(f'NR -> {nr}')
        # print(f'NO RECUPERADOS RELEVANTES -> {len(nr)}')

        return len(rr) / (len(rr) + len(nr))

    def compute_f1(self, relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
        precission = self.compute_precission(
            relevant_indexes, model_relevant_indexes)
        recall = self.compute_recall(
            relevant_indexes, model_relevant_indexes)

        if precission == recall == 0:
            return 0

        return 2 * (precission * recall) / (precission + recall)

    def average(self, values: List[float]) -> float:
        total = len(values)
        return round(sum(values) / total, 5)
