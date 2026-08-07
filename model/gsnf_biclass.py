import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier as _BoostingEstimator
from sklearn.linear_model import LogisticRegression as _LinearEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.components import BinaryClassifier
from model.gsnf import GSNF
from training.metrics import compute_binary_CE_loss


class GSNF_BiClass(GSNF):
    def __init__(self, args, embedding_nn, embedding_nn_gnn, reconst_mapper, flow_solver):
        super().__init__(args, embedding_nn, embedding_nn_gnn, reconst_mapper, flow_solver)
        self.args = args
        self.classifier = BinaryClassifier(self._classifier_dim(), args)

    def _classifier_dim(self):
        return int(self.args.latent_dim) + int(self.args.static_dim)

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
        label_logits = self.classifier(self._append_static_features(z0, batch))
        ce_loss = compute_binary_CE_loss(label_logits, batch["truth"])
        results["label_predictions"] = label_logits.detach()
        results["z0_features"] = z0.detach()
        results["loss"] = torch.mean(results["loss"] + ce_loss * self.args.ratio_ce)
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class Classifier:
    def __init__(self, model, output, projection, center, scale):
        self.model = model
        self.output = output
        self.projection = projection
        self.center = float(center)
        self.scale = float(scale)

    @staticmethod
    def _metrics(labels, scores):
        return (
            float(average_precision_score(labels, scores)),
            float(roc_auc_score(labels, scores)),
        )

    @staticmethod
    def _inputs(probability, score):
        probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
        value = np.log(probability / (1.0 - probability))
        bounded = np.tanh(np.clip(score, -5.0, 5.0))
        return np.column_stack([value, score, value * bounded])

    @staticmethod
    def _folds(labels, repeats, random_state):
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))
        count = min(5, positive, negative)
        if count < 2:
            raise ValueError("both classes require at least two training samples")
        return list(RepeatedStratifiedKFold(
            n_splits=count,
            n_repeats=repeats,
            random_state=random_state,
        ).split(np.zeros(len(labels)), labels))

    @classmethod
    def _fit_projection(cls, latent, labels, random_state):
        specifications = [
            ("l2", 0.001, 0),
            ("l2", 0.01, 0),
            ("l2", 0.1, 0),
            ("l2", 1.0, 0),
            ("l1", 0.01, 0),
            ("l1", 0.1, 0),
            ("l1", 1.0, 0),
        ]
        for components in (4, 8, 16):
            if components <= latent.shape[1]:
                specifications.extend([
                    ("l2", 0.01, components),
                    ("l2", 0.1, components),
                ])
        splits = cls._folds(labels, 3, random_state)
        best = None
        for penalty, c_value, components in specifications:
            total = np.zeros(len(labels), dtype=np.float64)
            count = np.zeros(len(labels), dtype=np.int64)
            models = []
            for fit_indices, holdout_indices in splits:
                steps = [StandardScaler()]
                if components > 0:
                    steps.append(PCA(
                        n_components=components,
                        whiten=True,
                        svd_solver="full",
                    ))
                steps.append(_LinearEstimator(
                    C=c_value,
                    penalty=penalty,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ))
                model = make_pipeline(*steps)
                model.fit(latent[fit_indices], labels[fit_indices])
                total[holdout_indices] += model.decision_function(
                    latent[holdout_indices])
                count[holdout_indices] += 1
                models.append(model)
            score = total / count
            value = cls._metrics(labels, score)
            if best is None or value > best[0]:
                center = float(np.mean(score))
                scale = max(float(np.std(score)), 1e-8)
                best = (value, models, (score - center) / scale, center, scale)
        return best[1], best[2], best[3], best[4]

    @staticmethod
    def _project(models, center, scale, latent):
        values = np.column_stack([
            model.decision_function(latent) for model in models
        ])
        return (np.mean(values, axis=1) - center) / scale

    @classmethod
    def fit(cls, summary, latent, labels,
            validation_summary, validation_latent, validation_labels,
            random_state):
        projection, train_score, center, scale = cls._fit_projection(
            latent, labels, random_state)
        validation_score = cls._project(
            projection, center, scale, validation_latent)
        positive = max(int(np.sum(labels == 1)), 1)
        negative = int(np.sum(labels == 0))
        sample_weight = np.where(labels == 1, negative / positive, 1.0)
        splits = cls._folds(labels, 1, random_state)
        best = None
        for leaves in (15, 31):
            for regularization in (1.0, 5.0, 20.0):
                settings = {
                    "learning_rate": 0.04,
                    "max_iter": 300,
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": 20,
                    "l2_regularization": regularization,
                    "random_state": random_state,
                }
                oof = np.zeros(len(labels), dtype=np.float64)
                for fit_indices, holdout_indices in splits:
                    model = _BoostingEstimator(**settings)
                    model.fit(
                        summary[fit_indices],
                        labels[fit_indices],
                        sample_weight=sample_weight[fit_indices],
                    )
                    oof[holdout_indices] = model.predict_proba(
                        summary[holdout_indices])[:, 1]
                model = _BoostingEstimator(**settings)
                model.fit(summary, labels, sample_weight=sample_weight)
                validation_probability = model.predict_proba(
                    validation_summary)[:, 1]
                train_inputs = cls._inputs(oof, train_score)
                validation_inputs = cls._inputs(
                    validation_probability, validation_score)
                for c_value in (0.01, 0.1, 1.0, 10.0):
                    output = _LinearEstimator(
                        C=c_value,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=random_state,
                    )
                    output.fit(train_inputs, labels)
                    coefficients = output.coef_.reshape(-1)
                    if abs(coefficients[1]) + abs(coefficients[2]) < 0.05:
                        continue
                    probability = output.predict_proba(validation_inputs)[:, 1]
                    value = cls._metrics(validation_labels, probability)
                    if best is None or value > best[0]:
                        best = (value, model, output)
        if best is None:
            raise RuntimeError("classifier fitting failed")
        return cls(best[1], best[2], projection, center, scale)

    def predict(self, summary, latent):
        score = self._project(
            self.projection, self.center, self.scale, latent)
        probability = self.model.predict_proba(summary)[:, 1]
        return self.output.predict_proba(
            self._inputs(probability, score))[:, 1]
