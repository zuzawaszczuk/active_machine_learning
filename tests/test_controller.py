from src.dataset import imbalance_train_dataset


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


def test_reset_clears_labeled_and_restores_unlabeled(controller, dataset):
    samples_to_label = [0, 5, 10, 15, 20]
    controller.label_samples(samples_to_label)

    assert len(controller.labeled_idx) == len(samples_to_label)
    assert len(controller.unlabeled_idx) == len(dataset) - len(samples_to_label)

    controller.reset()

    assert len(controller.labeled_idx) == 0
    assert len(controller.unlabeled_idx) == len(dataset)
    assert controller.unlabeled_idx == set(range(len(dataset)))


def test_get_labeled_loader_returns_correct_data(controller, dataset):
    samples_to_label = [0, 5, 10, 15, 20]
    controller.label_samples(samples_to_label)

    labeled_loader = controller.get_labeled_loader(batch_size=5)

    all_indices = []
    for batch_x, batch_y, batch_idx in labeled_loader:
        all_indices.extend(batch_idx.tolist())

    assert set(all_indices) == set(samples_to_label)
    assert len(all_indices) == len(samples_to_label)

    batch_x, batch_y, batch_idx = next(iter(labeled_loader))
    assert batch_x.shape[0] <= 5
    assert batch_x.shape == (batch_y.shape[0], 3, 32, 32)


def test_get_unlabeled_loader_returns_correct_data(controller, dataset):
    samples_to_label = [0, 5, 10, 15, 20]
    controller.label_samples(samples_to_label)

    unlabeled_loader = controller.get_unlabeled_loader(batch_size=10)

    all_indices = []
    for batch_x, batch_y, batch_idx in unlabeled_loader:
        all_indices.extend(batch_idx.tolist())

    assert len(set(all_indices) & set(samples_to_label)) == 0
    assert len(all_indices) == len(dataset) - len(samples_to_label)

    batch_x, batch_y, batch_idx = next(iter(unlabeled_loader))
    assert batch_x.shape[0] <= 10
    assert batch_x.shape == (batch_y.shape[0], 3, 32, 32)


def test_imbalance_train_dataset_creates_imbalanced_dataset_correctly(dataset):
    original_size = len(dataset)
    compression = 2

    imbalanced_dataset = imbalance_train_dataset(dataset, compression, num_classes=5)

    assert len(imbalanced_dataset) < original_size
    assert len(imbalanced_dataset) > 0
