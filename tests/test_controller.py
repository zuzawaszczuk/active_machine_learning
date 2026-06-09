import pytest
from src.controller import ActiveLearningController
from src.dataset import IndexedSubset


def test_label_samples_and_check_lengths(controller, dataset):
    assert len(controller.labeled_idx) == 0
    assert len(controller.unlabeled_idx) == len(dataset)
    
    samples_to_label = [0, 5, 10, 15, 20]
    controller.label_samples(samples_to_label)
    
    assert len(controller.labeled_idx) == len(samples_to_label)
    assert len(controller.unlabeled_idx) == len(dataset) - len(samples_to_label)
    
    assert controller.labeled_idx == set(samples_to_label)
    
    for idx in samples_to_label:
        assert idx not in controller.unlabeled_idx