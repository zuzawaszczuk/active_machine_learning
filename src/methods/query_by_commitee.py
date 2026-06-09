import numpy as np
from sklearn.ensemble import RandomForestClassifier


class QueryByCommittee:
    def __init__(
        self,
        n_estimators=100,
        score="vote_entropy",
        max_depth=None,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.score = score
        self.max_depth = max_depth
        self.random_state = random_state

    def select_samples(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled,
        unlabeled_indices,
        n_samples,
    ):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        rf.fit(X_labeled, y_labeled)

        if self.score == "vote_entropy":
            scores = self._vote_entropy(rf, X_unlabeled)

        elif self.score == "kl_divergence":
            scores = self._kl_divergence(rf, X_unlabeled)

        elif self.score == "variation_ratio":
            scores = self._variation_ratio(rf, X_unlabeled)

        else:
            raise ValueError(
                "score must be one of: vote_entropy, kl_divergence, variation_ratio"
            )

        selected_positions = np.argsort(scores)[-n_samples:]
        selected_indices = [unlabeled_indices[i] for i in selected_positions]

        return selected_indices

    def _vote_entropy(self, rf, X_unlabeled):
        tree_predictions = np.array([
            tree.predict(X_unlabeled)
            for tree in rf.estimators_
        ])

        tree_predictions = tree_predictions.T

        scores = []

        for votes in tree_predictions:
            labels, counts = np.unique(votes, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            scores.append(entropy)

        return np.array(scores)

    def _variation_ratio(self, rf, X_unlabeled):
        tree_predictions = np.array([
            tree.predict(X_unlabeled)
            for tree in rf.estimators_
        ])

        tree_predictions = tree_predictions.T

        scores = []

        for votes in tree_predictions:
            labels, counts = np.unique(votes, return_counts=True)
            max_vote_fraction = counts.max() / counts.sum()
            variation_ratio = 1 - max_vote_fraction
            scores.append(variation_ratio)

        return np.array(scores)

    def _kl_divergence(self, rf, X_unlabeled):
        all_probas = []

        for tree in rf.estimators_:
            proba = tree.predict_proba(X_unlabeled)
            all_probas.append(proba)

        all_probas = np.array(all_probas)

        mean_proba = np.mean(all_probas, axis=0)

        kl = all_probas * (
            np.log(all_probas + 1e-12)
            - np.log(mean_proba[None, :, :] + 1e-12)
        )

        scores = kl.sum(axis=2).mean(axis=0)

        return scores