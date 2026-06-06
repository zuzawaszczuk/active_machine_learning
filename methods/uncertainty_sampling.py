from abc import ABC, abstractmethod
from dataset import DatasetWithoutLabels


class UncertaintySampling(ABC):
    def __init__(self, dataset: DatasetWithoutLabels):
        self.dataset = dataset

    @abstractmethod
    def select_samples(self, model, X_pool, n_samples):
        pass

    def train(self, model, X_train, y_train, batch_size=32,epochs=10):
        
        
        model.fit(X_train, y_train)

    