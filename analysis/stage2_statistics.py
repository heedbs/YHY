"""Historical analysis functions; bootstrap entry adds explicit input guards.

No analysis runs on import. The historical reference script remains unchanged.
"""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

CLASSES = ['normal', 'osmf', 'oscc']
def metric_key(d):
    d = d.get("inner", d)
    auroc = d.get("macro_ovr_auroc")
    auroc = -float("inf") if auroc is None or not np.isfinite(auroc) else float(auroc)
    return (float(d["macro_f1"]), auroc, -float(d["loss"]), -int(d["epoch"]))

def brier(y, p):
    one = np.eye(3)[y]
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))

def metrics(y, p):
    pred = p.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_ovr_auroc": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "macro_ovr_auprc": float(average_precision_score(np.eye(3)[y], p, average="macro")),
        "multiclass_brier": brier(y, p),
    }
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        out[f"{c}_sensitivity"] = float(tp / (tp + fn)) if tp + fn else np.nan
        out[f"{c}_specificity"] = float(tn / (tn + fp)) if tn + fp else np.nan
        out[f"{c}_precision"] = float(tp / (tp + fp)) if tp + fp else np.nan
    return out, cm, pred

def bootstrap_metrics(y, p, idx):
    """Historical vectorized calculation, limited to supported stratified inputs.

Opposing-label OVR score ties are not handled by the historical rank/AP
algorithm; reject them rather than silently generalizing it to new datasets.
"""
    y=np.asarray(y);p=np.asarray(p);idx=np.asarray(idx)
    if y.ndim!=1 or p.shape!=(len(y),3) or not np.issubdtype(y.dtype,np.integer):
        raise ValueError('expected integer labels and N by 3 probabilities')
    if set(y.tolist())!={0,1,2} or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('three classes and finite probabilities in [0,1] required')
    if not np.allclose(p.sum(1),1,rtol=0,atol=1e-5):
        raise ValueError('probability row sums invalid')
    if idx.ndim!=2 or len(idx)==0 or idx.shape[1]!=len(y) or not np.issubdtype(idx.dtype,np.integer):
        raise ValueError('expected nonempty integer resampling indices with N columns')
    if (idx<0).any() or (idx>=len(y)).any():
        raise ValueError('resampling index out of range')
    for c in range(3):
        if not np.all((y[idx]==c).sum(1)==(y==c).sum()):
            raise ValueError('class-stratified patient counts must be preserved')
        order=np.argsort(p[:,c]);score=p[order,c];binary=y[order]==c
        if np.any((score[1:]==score[:-1]) & (binary[1:]!=binary[:-1])):
            raise ValueError('opposing-label OVR score ties unsupported by historical bootstrap implementation')
    yy = y[idx]
    pp = p[idx]
    pred = pp.argmax(2)
    n = len(idx)
    cm = np.zeros((n, 3, 3), dtype=np.int32)
    for t in range(3):
        for q in range(3):
            cm[:, t, q] = ((yy == t) & (pred == q)).sum(1)
    diag = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    row = cm.sum(2).astype(float); col = cm.sum(1).astype(float)
    sensitivity = diag / row
    precision = np.divide(diag, col, out=np.zeros_like(diag), where=col != 0)
    specificity = np.empty_like(sensitivity)
    for c in range(3):
        tn = cm.sum((1,2)) - row[:,c] - col[:,c] + diag[:,c]
        specificity[:,c] = tn / (tn + col[:,c] - diag[:,c])
    f1 = np.divide(2 * precision * sensitivity, precision + sensitivity, out=np.zeros_like(precision), where=(precision+sensitivity)!=0)
    out = {
        "macro_f1": f1.mean(1),
        "balanced_accuracy": sensitivity.mean(1),
        "multiclass_brier": ((pp - np.eye(3)[yy]) ** 2).sum(2).mean(1),
    }
    for i,c in enumerate(CLASSES):
        out[f"{c}_sensitivity"] = sensitivity[:,i]
        out[f"{c}_specificity"] = specificity[:,i]
        out[f"{c}_precision"] = precision[:,i]
    aurocs=[]; aps=[]
    for c in range(3):
        target=(yy==c).astype(np.int8); score=pp[:,:,c]
        order=np.argsort(score,axis=1)
        rank=np.empty_like(order); rank[np.arange(n)[:,None],order]=np.arange(1,score.shape[1]+1)
        npos=target.sum(1).astype(float); nneg=score.shape[1]-npos
        aurocs.append(((rank*target).sum(1)-npos*(npos+1)/2)/(npos*nneg))
        desc=np.argsort(-score,axis=1); sorted_y=np.take_along_axis(target,desc,axis=1)
        precision_at=np.cumsum(sorted_y,axis=1)/np.arange(1,score.shape[1]+1)
        aps.append((precision_at*sorted_y).sum(1)/npos)
    out["macro_ovr_auroc"]=np.mean(aurocs,axis=0)
    out["macro_ovr_auprc"]=np.mean(aps,axis=0)
    return out
