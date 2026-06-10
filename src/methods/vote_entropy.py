
from .uncertainty_sampling import UncertaintySampling
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class MarginSampling(UncertaintySampling):
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

                top2 = probs.topk(k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]

                scores.extend(margin.cpu().tolist())
                pool_indices.extend(idx.tolist())

        _, topk_idx = torch.topk(torch.tensor(scores), n_samples, largest=False)
        selected = [pool_indices[i] for i in topk_idx.tolist()]

        return selected
