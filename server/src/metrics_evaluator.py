from typing import List, Set



def compute_precission(cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
    if len(cran_relevant_indexes) == 0:
        return 0

    cran_set = set(cran_relevant_indexes)
    model_set = set(model_relevant_indexes)
    rr = cran_set.intersection(model_set)
    ri = model_set.difference(cran_set)
    return len(rr) / (len(rr) + len(ri))


def compute_recall(cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
    if len(cran_relevant_indexes) == 0:
        return 0

    cran_set = set(cran_relevant_indexes)
    model_set = set(model_relevant_indexes)
    rr = cran_set.intersection(model_set)
    nr = cran_set.difference(model_set)
    return len(rr) / (len(rr) + len(nr))


def compute_f1(cran_relevant_indexes: List[int], model_relevant_indexes: List[int]) -> float:
    precission = compute_precission(
        cran_relevant_indexes, model_relevant_indexes)
    recall = compute_recall(cran_relevant_indexes, model_relevant_indexes)

    if precission == recall == 0:
        return 0

    return (2 / (1/precission) + (1/recall))
