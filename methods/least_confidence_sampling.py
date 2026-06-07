from .uncertainty_sampling import UncertaintySampling
import torch.nn as nn
from torch.utils.data import DataLoader
import torch


class LeastConfidenceSampling(UncertaintySampling):
    def select_samples(
        self, model: nn.Module, X_pool: DataLoader, n_samples: int, device: torch.device
    ) -> list[int]:
        model.eval()

        scores = []
        pool_indices = []

        with torch.no_grad():
            for x, y, idx in X_pool:
                x = x.to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)

                confidence = probs.max(dim=1).values
                uncertainty = 1 - confidence

                scores.extend(uncertainty.cpu().tolist())
                pool_indices.extend(idx.tolist())

        _, topk_idx = torch.topk(torch.tensor(scores), n_samples)
        selected = [pool_indices[i] for i in topk_idx.tolist()]

        return selected
