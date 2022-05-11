from src.dataset import Dataset


class VectorSpaceModel:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
