from .uncertainty_sampling import UncertaintySampling
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class LeastConfidenceSampling(UncertaintySampling):
    def select_samples(self, model: nn.Module, X_pool: DataLoader, n_samples: int, device: torch.device) -> list[int]:
        model.eval()

        scores = []
        indices = []

        with torch.no_grad():
            for x, _ in X_pool:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

                confidence = probs.max(dim=1).values
                uncertainty = 1 - confidence

                scores.extend(uncertainty.cpu().tolist())

        _, indices = torch.topk(torch.tensor(scores), n_samples)
        return indices.tolist()