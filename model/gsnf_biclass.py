import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from model.components import BinaryClassifier
from model.gsnf import GSNF
from training.metrics import compute_binary_CE_loss


class GSNF_BiClass(GSNF):
    """GSNF with the fixed initial-state mortality classifier."""

    def __init__(self, args, embedding_nn, embedding_nn_gnn, reconst_mapper, flow_solver):
        super().__init__(args, embedding_nn, embedding_nn_gnn, reconst_mapper, flow_solver)
        self.args = args
        self.classifier = BinaryClassifier(self._classifier_dim(), args)

    def _classifier_dim(self):
        return int(self.args.latent_dim) + int(self.args.static_dim)

    def init_classifier_bias(self, positive_rate):
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            return
        last = self.classifier.layers[-1]
        last.bias.data.fill_(torch.logit(torch.tensor(positive_rate)).item())

    def _append_static_features(self, latent, batch):
        static = batch.get("static_in")
        if static is None or static.size(-1) != int(self.args.static_dim):
            raise ValueError("static_in must contain the configured static features")
        static = static.to(latent)
        static = static.unsqueeze(0).expand(latent.size(0), -1, -1)
        return torch.cat([latent, static], dim=-1)

    def compute_prediction_results(self, batch, k_iwae=1):
        results, forward_info = self.forward(batch, k_iwae)
        z0 = forward_info["initial_state"][:, :, 0, :]
        classifier_input = self._append_static_features(z0, batch)
        label_logits = self.classifier(classifier_input)
        ce_loss = compute_binary_CE_loss(
            label_logits,
            batch["truth"],
            pos_weight=getattr(self.args, "ce_pos_weight_runtime", 0.0),
        )
        results["label_predictions"] = label_logits.detach()
        results["z0_features"] = z0.detach()
        results["loss"] = torch.mean(results["loss"] + ce_loss * self.args.ratio_ce)
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class ContextAdapter:
    """Validation-selected structured and neural risk-score fusion."""

    def __init__(self, model, blend_weight):
        self.model = model
        self.blend_weight = float(blend_weight)

    @classmethod
    def fit(cls, train_summary, train_logits, train_labels,
            validation_summary, validation_logits, validation_labels,
            random_state):
        positive = max(int(np.sum(train_labels == 1)), 1)
        negative = int(np.sum(train_labels == 0))
        sample_weight = np.where(train_labels == 1, negative / positive, 1.0)
        validation_neural = 1.0 / (1.0 + np.exp(-validation_logits))
        best = None
        for leaves in (15, 31):
            for l2 in (1.0, 5.0, 20.0):
                model = HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=300,
                    max_leaf_nodes=leaves,
                    min_samples_leaf=20,
                    l2_regularization=l2,
                    random_state=random_state,
                )
                model.fit(train_summary, train_labels, sample_weight=sample_weight)
                tree_score = model.predict_proba(validation_summary)[:, 1]
                for weight in (1e-6, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35):
                    score = (1.0 - weight) * tree_score + weight * validation_neural
                    value = (
                        float(average_precision_score(validation_labels, score)),
                        float(roc_auc_score(validation_labels, score)),
                    )
                    if best is None or value > best[0]:
                        best = (value, model, weight)
        return cls(best[1], best[2])

    def predict(self, summary, neural_logits):
        tree_score = self.model.predict_proba(summary)[:, 1]
        neural_score = 1.0 / (1.0 + np.exp(-neural_logits))
        return (1.0 - self.blend_weight) * tree_score + self.blend_weight * neural_score
