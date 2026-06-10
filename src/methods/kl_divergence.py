import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from .uncertainty_sampling import UncertaintySampling


class KLDivergenceSampling(UncertaintySampling):
    def select_samples(
        self,
        model: RandomForestClassifier,
        X_pool: DataLoader,
        n_samples: int,
        device: torch.device,
    ) -> list[int]:
        scores = []
        pool_indices = []

        classes = model.classes_

        for x, y, idx in X_pool:
            x = x.view(x.size(0), -1)
            x = x.cpu().numpy()
            all_probas = []

            for tree in model.estimators_:
                tree_proba = tree.predict_proba(x)
                tree_classes = tree.classes_

                aligned_proba = np.zeros((x.shape[0], len(classes)))

                for i, cls in enumerate(tree_classes):
                    class_idx = np.where(classes == cls)[0][0]
                    aligned_proba[:, class_idx] = tree_proba[:, i]

                all_probas.append(aligned_proba)

            all_probas = np.array(all_probas)
            mean_proba = np.mean(all_probas, axis=0)

            kl = all_probas * (
                np.log(all_probas + 1e-12)
                - np.log(mean_proba[None, :, :] + 1e-12)
            )

            batch_scores = kl.sum(axis=2).mean(axis=0)

            scores.extend(batch_scores.tolist())
            pool_indices.extend(idx.tolist())

        _, topk_idx = torch.topk(torch.tensor(scores), n_samples, largest=True)
        selected = [pool_indices[i] for i in topk_idx.tolist()]

        return selected