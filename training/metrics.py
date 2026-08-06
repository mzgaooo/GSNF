import torch
import torch.nn.functional as F


def compute_binary_CE_loss(
        label_predictions,
        mortality_label,
        pos_weight=0.0,
        label_smoothing=0.0,
        focal_gamma=0.0):
    """Binary classification objective for the mortality head."""
    labels = mortality_label.reshape(-1)
    logits = label_predictions
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = logits.reshape(logits.size(0), -1)
    valid = ~torch.isnan(labels)
    logits = logits[:, valid]
    labels = labels[valid]
    if logits.numel() == 0:
        return label_predictions.new_zeros(())
    labels = labels.unsqueeze(0).expand_as(logits)

    if label_smoothing > 0.0:
        smooth = float(label_smoothing)
        labels = labels * (1.0 - smooth) + 0.5 * smooth

    weight = None
    if pos_weight is not None and float(pos_weight) > 0.0:
        weight = label_predictions.new_tensor([float(pos_weight)])

    loss = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=weight, reduction="none")
    if focal_gamma > 0.0:
        probability = torch.sigmoid(logits)
        true_probability = (
            probability * labels + (1.0 - probability) * (1.0 - labels))
        loss = loss * (1.0 - true_probability).pow(float(focal_gamma))
    return loss.mean()


def log_normal_pdf(x, mean, logvar, mask):
    log_two_pi = x.new_tensor(6.283185307179586).log()
    return -0.5 * (
        log_two_pi + logvar + (x - mean).pow(2) / torch.exp(logvar)
    ) * mask


def compute_log_normal_pdf(observed_data, observed_mask, pred_x, args):
    obsrv_std = torch.zeros_like(pred_x) + args.obsrv_std
    noise_logvar = 2.0 * torch.log(obsrv_std)
    pdf = log_normal_pdf(observed_data, pred_x, noise_logvar, observed_mask)
    logpx = pdf.sum(-1).sum(-1)
    if args.norm:
        logpx = logpx / observed_mask.sum(-1).sum(-1).clamp_min(1.0)
    return logpx
