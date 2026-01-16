# scripts/metrics.py
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm, eps=1e-12):
    K = cm.shape[0]
    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0).astype(np.float64) - TP
    FN = cm.sum(axis=1).astype(np.float64) - TP
    TN = cm.sum().astype(np.float64) - (TP + FP + FN)

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)  # sensitivity
    f1 = 2 * precision * recall / (precision + recall + eps)
    specificity = TN / (TN + FP + eps)  # one-vs-rest per class
    return precision, recall, f1, specificity


def accuracy_from_cm(cm):
    return float(np.trace(cm) / (cm.sum() + 1e-12))


def cohen_kappa(cm):
    # Multiclass Cohen's kappa
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = (row @ col) / (n * n)
    if (1 - pe) == 0:
        return 0.0
    return float((po - pe) / (1 - pe))


def multiclass_mcc(cm):
    # Multiclass MCC (Wikipedia form)
    s = cm.sum().astype(np.float64)
    if s == 0:
        return 0.0
    c = np.trace(cm).astype(np.float64)
    p_k = cm.sum(axis=0).astype(np.float64)  # predicted totals (cols)
    t_k = cm.sum(axis=1).astype(np.float64)  # true totals (rows)

    num = c * s - np.sum(p_k * t_k)
    den = np.sqrt((s * s - np.sum(p_k * p_k)) * (s * s - np.sum(t_k * t_k)))
    if den == 0:
        return 0.0
    return float(num / den)


def summarize_metrics(cm):
    prec, rec, f1, spec = per_class_prf(cm)
    acc = accuracy_from_cm(cm)

    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    macro_spec = float(np.mean(spec))

    kappa = cohen_kappa(cm)
    mcc = multiclass_mcc(cm)

    return {
        "acc": acc,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_specificity": spec,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_specificity": macro_spec,
        "kappa": kappa,
        "mcc": mcc,
    }
