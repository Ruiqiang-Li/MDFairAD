import torch
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F


def eval_metric(output, labels, sens, idx, args):
    loss = F.binary_cross_entropy_with_logits(output[idx],
                                              labels[idx].unsqueeze(1).float().to(args.device))

    output_preds = (output.squeeze() > 0).type_as(labels)

    auc_roc = roc_auc_score(labels[idx].cpu().numpy(),
                            output[idx].detach().cpu().numpy())
    parity, equality = fair_metric(output_preds[idx].cpu().numpy(),
                                   labels[idx].cpu().numpy(),
                                   sens[idx].cpu().numpy())

    f1 = f1_score(labels[idx].cpu().numpy(),
                  output_preds[idx].cpu().numpy())
    accuracy = output_preds[idx].eq(labels[idx]).sum().item() / idx.shape[0]

    return loss, auc_roc, parity, equality, f1, accuracy


def select_best_threshold(output, labels, sens, idx):
    logits = output[idx].detach().cpu().numpy().reshape(-1)
    labels_np = labels[idx].cpu().numpy()
    sens_np = sens[idx].cpu().numpy()

    candidate_thresholds = np.array([
        -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0
    ], dtype=np.float32)

    unique_logits = np.unique(logits.astype(np.float32))
    if unique_logits.size > 1:
        midpoint_thresholds = ((unique_logits[:-1] + unique_logits[1:]) / 2.0).astype(np.float32)
        if midpoint_thresholds.size > 63:
            midpoint_percentiles = np.linspace(0, 100, 63, dtype=np.float32)
            midpoint_thresholds = np.percentile(midpoint_thresholds, midpoint_percentiles).astype(np.float32)
        candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, midpoint_thresholds]))

    auc_roc = roc_auc_score(labels_np, logits)
    target_positive_rate = float(labels_np.mean())
    best_threshold = 0.0
    best_metrics = None
    best_score = float('-inf')
    best_tuple = None

    for threshold in candidate_thresholds:
        preds = (logits > threshold).astype(labels_np.dtype)
        positive_rate = float(preds.mean())
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            continue

        parity, equality = fair_metric(preds, labels_np, sens_np)
        f1 = f1_score(labels_np, preds)
        accuracy = float((preds == labels_np).mean())
        score = auc_roc + f1 - (parity + equality)
        tie_break = (f1, accuracy, -(parity + equality), -abs(positive_rate - target_positive_rate))

        if score > best_score or (score == best_score and (best_tuple is None or tie_break > best_tuple)):
            best_score = score
            best_tuple = tie_break
            best_threshold = float(threshold)
            best_metrics = (parity, equality, f1, accuracy)

    if best_metrics is not None:
        return best_threshold, *best_metrics

    preds = (logits > 0.0).astype(labels_np.dtype)
    parity, equality = fair_metric(preds, labels_np, sens_np)
    f1 = f1_score(labels_np, preds)
    accuracy = float((preds == labels_np).mean())
    return 0.0, parity, equality, f1, accuracy


def eval_metric_with_threshold(output, labels, sens, idx, args, threshold):
    loss = F.binary_cross_entropy_with_logits(output[idx],
                                              labels[idx].unsqueeze(1).float().to(args.device))

    logits = output[idx].detach().cpu().numpy().reshape(-1)
    labels_np = labels[idx].cpu().numpy()
    sens_np = sens[idx].cpu().numpy()
    preds = (logits > threshold).astype(labels_np.dtype)

    auc_roc = roc_auc_score(labels_np, logits)
    parity, equality = fair_metric(preds, labels_np, sens_np)
    f1 = f1_score(labels_np, preds)
    accuracy = float((preds == labels_np).mean())

    return loss, auc_roc, parity, equality, f1, accuracy


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def metric_wd(feature, flag, plt_show=False):
    flag = flag.detach().cpu()
    feature = (feature / feature.norm(dim=0)).detach().cpu().numpy()
    emd_distances = []

    for i in range(feature.shape[1]):
        class_1 = feature[torch.eq(flag, 0), i]
        class_2 = feature[torch.eq(flag, 1), i]
        emd = wasserstein_distance(class_1, class_2)
        emd_distances.append(emd)

    if plt_show:
        print('Attribute bias : ')
        print("Sum of all Wasserstein distance value across feature dimensions: " + str(sum(emd_distances)))
        print(
            "Average of all Wasserstein distance value across feature dimensions: " + str(
                np.mean(np.array(emd_distances))))

        sns.distplot(np.array(emd_distances).squeeze(), rug=True, hist=True, label='EMD value distribution')
        plt.legend()
        plt.show()

        num_list1 = emd_distances.cpu().numpy()
        x = range(len(num_list1))

        plt.bar(x, height=num_list1, width=0.4, alpha=0.8, label="Wasserstein distance on reachability")
        plt.ylabel("Wasserstein distance")
        plt.legend()
        plt.show()

    return sum(emd_distances)
