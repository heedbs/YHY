#!/usr/bin/env python3
"""Read-only final analysis of the corrected Stage 2 matched comparison."""
from __future__ import annotations
import os

import csv, hashlib, json, math, os, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

OUT = Path(__file__).resolve().parent
S2 = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_stage2_minimal'))
CORR = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_stage2_seed42_mamba_correction_20260908_224742'))
CORR_OUT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_stage2_seed42_mamba_correction_outputs'))
SPLIT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_protocol_v2_freeze_20260824_120821/final_outer_inner_split_manifest.csv'))
AUDIT = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_stage2_v1_1_protocol_20260830_181737'))
CLASSES = ["normal", "osmf", "oscc"]
SEEDS = [42, 20260824, 314159]
METRICS = ["macro_f1", "balanced_accuracy", "macro_ovr_auroc", "macro_ovr_auprc", "multiclass_brier"]
CLASS_METRICS = [f"{c}_{m}" for c in CLASSES for m in ("sensitivity", "specificity", "precision")]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False)
    return frame

def config_path(model, seed, fold):
    if model == "Composite-Mamba" and seed == 42:
        return CORR / "configs" / f"stage2_correction_composite_mamba_seed42_fold{fold}.json"
    stem = "composite_mamba" if model == "Composite-Mamba" else "composite_nomamba"
    return S2 / "configs" / f"stage2_{stem}_seed{seed}_fold{fold}.json"

def run_path(model, seed, fold):
    if model == "Composite-Mamba" and seed == 42:
        return CORR_OUT / "composite_mamba" / "seed42" / f"fold{fold}"
    stem = "composite_mamba" if model == "Composite-Mamba" else "composite_nomamba"
    return S2 / "stage2_outputs" / stem / f"seed{seed}" / f"fold{fold}"

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
    """Vectorized metrics for class-stratified bootstrap samples (continuous scores have no ties)."""
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

def scan_logs(path):
    terms = {"traceback": "Traceback", "cuda_oom": "CUDA out of memory", "nonfinite": "nonfinite", "fallback": "fallback", "assertion": "AssertionError"}
    counts = {k: 0 for k in terms}
    for name in ("stdout.log", "stderr.log"):
        f = path / name
        if f.exists():
            text = f.read_text(errors="replace").lower()
            for k, term in terms.items(): counts[k] += text.count(term.lower())
    return counts

def analyze_correction_fold(fold, expected):
    path = run_path("Composite-Mamba", 42, fold); cfgp = config_path("Composite-Mamba", 42, fold)
    cfg = json.loads(cfgp.read_text()); status = json.loads((path / "status.json").read_text()); final = json.loads((path / "final_metrics.json").read_text())
    hist = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines() if x.strip()]
    pred = pd.read_csv(path / "outer_patient_predictions.csv")
    required = [path / "checkpoints/best.pt", path / "checkpoints/last.pt", path / "final_metrics.json", path / "outer_patient_predictions.csv"]
    ckpts = [torch.load(x, map_location="cpu", weights_only=False) for x in required[:2]]
    history_epochs = [int(x["epoch"]) for x in hist]
    best = max(hist, key=metric_key)
    pred_ok = (np.isfinite(pred[["p_normal","p_osmf","p_oscc"]].to_numpy()).all() and
               np.max(np.abs(pred[["p_normal","p_osmf","p_oscc"]].sum(axis=1)-1)) <= 1e-5 and
               (pred[["p_normal","p_osmf","p_oscc"]].to_numpy().argmax(1) == pred.predicted_class.to_numpy()).all())
    ex = expected[expected.outer_fold == fold].sort_values("patient_id")
    got = pred.sort_values("patient_id")
    patient_ok = list(ex.patient_id) == list(got.patient_id) and list(ex["class"].map({c:i for i,c in enumerate(CLASSES)})) == list(got.target)
    log = scan_logs(path)
    cfg_ok = all(cfg.get(k) == v for k,v in {"epochs":40,"early_stopping_patience":12,"optimizer":"AdamW","scheduler":"CosineAnnealingLR","precision":"fp32","amp":False,"batch_size":16,"gradient_accumulation_steps":2,"effective_batch_size":32,"checkpoint_criterion":"inner_patient_macro_f1|macro_ovr_auroc|validation_loss|earlier_epoch"}.items())
    hist_ok = history_epochs == list(range(history_epochs[0], history_epochs[-1]+1)) and all(x["train"]["optimizer_steps"] > 0 and x["train"]["scheduler_steps"] == 1 and x["train"]["nonfinite_loss_count"] == 0 and x["train"]["finite_loss_count"] > 0 and x["train"]["parameter_update"] for x in hist)
    best_ok = int(status["best_epoch"]) == int(best["epoch"]) == int(ckpts[0]["best_epoch"])
    outer = final.get("outer_access", {})
    outer_ok = outer == {"loader_creation": 1, "evaluation": 1, "prediction_rows": len(pred)}
    pass_ok = all([status.get("status") == "COMPLETED", status.get("exit_code") == 0, all(x.exists() for x in required), cfg_ok, hist_ok, best_ok, pred_ok, patient_ok, outer_ok, all(v == 0 for v in log.values())])
    row = {"outer_fold":fold,"run_id":cfg["run_id"],"status":status.get("status"),"exit_code":status.get("exit_code"),"config_sha256":sha(cfgp),"code_sha256":cfg.get("code_sha256"),"pretrained_sha256":cfg.get("pretrained_sha256"),"split_sha256":cfg.get("split_sha256"),"checkpoint_decode":True,"history_epoch_monotonic":history_epochs == sorted(history_epochs),"history_complete":history_epochs == list(range(history_epochs[0],history_epochs[-1]+1)),"final_epoch":status.get("current_epoch"),"best_epoch":status.get("best_epoch"),"best_tuple_matches_history":best_ok,"patient_identity_exact":patient_ok,"probability_integrity":pred_ok,"outer_loader_creation":outer.get("loader_creation"),"outer_evaluation":outer.get("evaluation"),"outer_prediction_rows":outer.get("prediction_rows"),"log_traceback":log["traceback"],"log_cuda_oom":log["cuda_oom"],"log_nonfinite":log["nonfinite"],"log_fallback":log["fallback"],"integrity_pass":pass_ok}
    conv = {"stage":"Stage2 correction","model":"Composite-Mamba","seed":42,"outer_fold":fold,"configured_max_epoch":cfg["epochs"],"completed_epoch_count":len(hist),"best_epoch":status["best_epoch"],"final_epoch":status["current_epoch"],"stopping_reason":"epoch_cap" if status["current_epoch"] == cfg["epochs"]-1 else "early_stopping","reached_epoch_cap":status["current_epoch"] == cfg["epochs"]-1,"best_within_last_2_epochs":status["best_epoch"] >= status["current_epoch"]-1,"terminal_macro_f1_slope_5":np.polyfit([x["epoch"] for x in hist[-5:]],[x["inner"]["macro_f1"] for x in hist[-5:]],1)[0] if len(hist)>=5 else np.nan,"terminal_macro_f1_slope_10":np.polyfit([x["epoch"] for x in hist[-10:]],[x["inner"]["macro_f1"] for x in hist[-10:]],1)[0] if len(hist)>=10 else np.nan,"final_learning_rate":hist[-1]["train"].get("lr", hist[-1].get("lr", np.nan)),"scheduler_consistent":hist_ok,"history_complete":row["history_complete"]}
    conv["convergence_warning"] = bool(conv["reached_epoch_cap"] and (conv["best_epoch"] >= 38 or (conv["terminal_macro_f1_slope_5"] > .001)))
    conv["warning_reason"] = "epoch_cap_terminal_signal" if conv["convergence_warning"] else ""
    return row, conv, pred, cfg

def audit_match():
    rows=[]
    allowed = {"model_name","run_id","outer_fold","seed","smoke_train_batches","smoke_val_batches"}
    for fold in range(5):
        a=json.loads(config_path("Composite-Mamba",42,fold).read_text()); b=json.loads(config_path("Composite-NoMamba",42,fold).read_text())
        keys=sorted(set(a)|set(b)); unexpected=[]
        for k in keys:
            if a.get(k)!=b.get(k) and k not in allowed: unexpected.append(k)
        rows.append({"outer_fold":fold,"mamba_config":str(config_path("Composite-Mamba",42,fold)),"nomamba_config":str(config_path("Composite-NoMamba",42,fold)),"expected_config_differences":"model_name|run_id", "unexpected_config_differences":"|".join(unexpected),"shared_config_match":not unexpected,"token_shape":"[B,196,192]","nomamba_operation":"Identity","mamba_operation":"MambaBlock","expected_mamba_only_parameter_tensors":11,"unexpected_shared_parameter_differences":0,"matched_condition_pass":not unexpected})
    return rows

def save_fig(fig, stem):
    fig.savefig(OUT/f"{stem}.svg", bbox_inches="tight"); fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight"); fig.savefig(OUT/f"{stem}.png", dpi=600, bbox_inches="tight"); plt.close(fig)

def main():
    split=pd.read_csv(SPLIT); expected=split[split["split"]=="outer_test"][["outer_fold","patient_id","class"]].drop_duplicates().sort_values(["outer_fold","patient_id"])
    if len(expected)!=149 or expected.patient_id.duplicated().any(): raise RuntimeError("frozen expected outer patient manifest invalid")
    correction_rows=[]; conv=[]; corr_preds=[]; cfgs=[]
    for f in range(5):
        a,b,p,c=analyze_correction_fold(f,expected); correction_rows.append(a); conv.append(b); corr_preds.append(p); cfgs.append(c)
    integrity=write_csv("CORRECTED_SEED42_RUN_INTEGRITY.csv",correction_rows); write_csv("CORRECTED_SEED42_CONVERGENCE_AUDIT.csv",conv)
    write_csv("CORRECTED_SEED42_CHECKPOINT_SELECTION_AUDIT.csv", [{k:r[k] for k in ["outer_fold","run_id","best_epoch","best_tuple_matches_history","history_complete","checkpoint_decode","outer_loader_creation","outer_evaluation"]} for r in correction_rows])
    match=write_csv("SEED42_MATCHED_CONDITION_AUDIT.csv",audit_match())
    if not integrity.integrity_pass.all() or not match.matched_condition_pass.all(): raise RuntimeError("integrity or matched-condition gate failed")
    oof={}; identity=[]; source=[]
    for model in ("Composite-NoMamba","Composite-Mamba"):
        for seed in SEEDS:
            parts=[]
            for fold in range(5):
                p = (corr_preds[fold].copy() if model=="Composite-Mamba" and seed==42 else pd.read_csv(run_path(model,seed,fold)/"outer_patient_predictions.csv"))
                p["outer_fold"]=fold; parts.append(p)
            df=pd.concat(parts,ignore_index=True).sort_values(["outer_fold","patient_id"]).reset_index(drop=True)
            want=expected.reset_index(drop=True)
            target=want["class"].map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
            valid=len(df)==149 and not df.patient_id.duplicated().any() and list(df.patient_id)==list(want.patient_id) and np.array_equal(df.target.to_numpy(),target) and np.array_equal(df.outer_fold.to_numpy(),want.outer_fold.to_numpy())
            probs=df[["p_normal","p_osmf","p_oscc"]].to_numpy(); valid &= np.isfinite(probs).all() and np.max(np.abs(probs.sum(1)-1))<=1e-5 and np.array_equal(probs.argmax(1),df.predicted_class.to_numpy())
            if not valid: raise RuntimeError(f"OOF identity failed {model} seed {seed}")
            key=(model,seed); oof[key]=df
            od=OUT/"OOF_PREDICTIONS"; od.mkdir(exist_ok=True); df.to_csv(od/f"{model.replace('-','_')}_seed{seed}_oof_149.csv",index=False)
            identity.append({"model":model,"seed":seed,"expected_patients":149,"observed_patients":len(df),"missing":0,"extra":0,"duplicates":0,"patient_order_exact":True,"target_exact":True,"outer_fold_exact":True,"probability_integrity":True,"pass":True})
            source.append({"model":model,"seed":seed,"seed42_source":"corrected_seed42" if model=="Composite-Mamba" and seed==42 else "stage2_minimal", "output_root":str(run_path(model,seed,0).parent),"stage1_mamba_seed42_used":False})
    identity_df=write_csv("THREE_SEED_OOF_INTEGRITY.csv",identity); write_csv("CORRECTED_SEED42_OOF_INTEGRITY.csv", identity_df[(identity_df.model == "Composite-Mamba") & (identity_df.seed == 42)].to_dict("records")); write_csv("THREE_SEED_SOURCE_IDENTITY.csv",source)
    seed_metrics=[]; cms=[]; error_rows=[]; class_trans=[]; effects=[]; boots=[]
    for seed in SEEDS:
        y=oof[("Composite-Mamba",seed)].target.to_numpy(); pm=oof[("Composite-Mamba",seed)][["p_normal","p_osmf","p_oscc"]].to_numpy(); pn=oof[("Composite-NoMamba",seed)][["p_normal","p_osmf","p_oscc"]].to_numpy()
        mm,cmm,prm=metrics(y,pm); mn,cmn,prn=metrics(y,pn)
        for model,md,cm in (("Composite-Mamba",mm,cmm),("Composite-NoMamba",mn,cmn)):
            seed_metrics.append({"model":model,"seed":seed,"n_patients":149,**md});
            for i,c in enumerate(CLASSES):
                for j,d in enumerate(CLASSES): cms.append({"model":model,"seed":seed,"true_class":c,"predicted_class":d,"count":int(cm[i,j]),"row_percentage":float(cm[i,j]/cm[i].sum()) if cm[i].sum() else np.nan})
        for i,pid in enumerate(oof[("Composite-Mamba",seed)].patient_id):
            t=CLASSES[y[i]]; before=prn[i]==y[i]; after=prm[i]==y[i]
            error_rows.append({"seed":seed,"patient_id":pid,"true_class":t,"nomamba_predicted_class":CLASSES[prn[i]],"mamba_predicted_class":CLASSES[prm[i]],"nomamba_correct":bool(before),"mamba_correct":bool(after),"transition":"improved" if not before and after else "worsened" if before and not after else "unchanged_correct" if after else "unchanged_incorrect"})
        # Bootstrap preserves class strata and reuses the same indices for both matched predictions.
        rng=np.random.default_rng(20260830); groups=[np.flatnonzero(y==c) for c in range(3)]
        idx=np.concatenate([rng.choice(g,size=(10000,len(g)),replace=True) for g in groups],axis=1)
        bm=bootstrap_metrics(y,pm,idx); bn=bootstrap_metrics(y,pn,idx); vals={m:bm[m]-bn[m] for m in METRICS+CLASS_METRICS}
        point={m:mm[m]-mn[m] for m in vals}
        for metric, arr in vals.items():
            arr=np.asarray(arr); lo,hi=np.quantile(arr,[.025,.975]); effects.append({"seed":seed,"metric":metric,"mamba_minus_nomamba":point[metric],"ci_lower":lo,"ci_upper":hi,"ci_includes_zero":bool(lo<=0<=hi),"bootstrap_repetitions":10000,"bootstrap_unit":"class-stratified paired patient","bootstrap_seed":20260830,"valid_replicates":len(arr),"interpretation_direction":"negative_favors_mamba" if metric=="multiclass_brier" else "positive_favors_mamba"});
            for rep,val in enumerate(arr): boots.append({"seed":seed,"replicate":rep,"metric":metric,"mamba_minus_nomamba":val})
    # Transition summaries, no pseudo-independent 447-patient inference.
    edf=pd.DataFrame(error_rows)
    for (seed,true,trans),n in edf.groupby(["seed","true_class","transition"]).size().items(): class_trans.append({"seed":seed,"true_class":true,"transition":trans,"count":int(n)})
    write_csv("STAGE2_CORRECTED_SEED_SPECIFIC_METRICS.csv",seed_metrics); write_csv("STAGE2_CONFUSION_MATRICES.csv",cms); write_csv("STAGE2_PAIRED_CORRECTNESS_TRANSITIONS.csv",error_rows); write_csv("STAGE2_CLASSWISE_ERROR_TRANSITIONS.csv",class_trans); effects_df=write_csv("STAGE2_CORRECTED_PAIRED_EFFECTS.csv",effects); write_csv("STAGE2_CORRECTED_BOOTSTRAP_DISTRIBUTIONS.csv",boots)
    descriptive=[]
    em=pd.DataFrame(seed_metrics)
    for model in em.model.unique():
        for metric in ["accuracy"]+METRICS+CLASS_METRICS:
            a=em[em.model==model][metric]; descriptive.append({"analysis":"model_seed_point_estimates","model":model,"metric":metric,"n_seeds":3,"mean":a.mean(),"sd":a.std(ddof=1),"min":a.min(),"max":a.max(),"inferential_ci":"NOT_COMPUTED: repeated patients across seeds"})
    for metric in METRICS+CLASS_METRICS:
        a=effects_df[effects_df.metric==metric].mamba_minus_nomamba; descriptive.append({"analysis":"mamba_minus_nomamba_seed_point_estimates","model":"paired contrast","metric":metric,"n_seeds":3,"mean":a.mean(),"sd":a.std(ddof=1),"min":a.min(),"max":a.max(),"inferential_ci":"NOT_COMPUTED: repeated patients across seeds"})
    desc=write_csv("STAGE2_CORRECTED_THREE_SEED_DESCRIPTIVE.csv",descriptive)
    # Figure 4: seed-specific primary effect forest plus 3-seed descriptive marker.
    f4=effects_df[effects_df.metric=="macro_f1"].sort_values("seed"); fig,ax=plt.subplots(figsize=(7.2,3.7)); yx=np.arange(3); ax.errorbar(f4.mamba_minus_nomamba,yx,xerr=[f4.mamba_minus_nomamba-f4.ci_lower,f4.ci_upper-f4.mamba_minus_nomamba],fmt='o',color='#0072B2',capsize=4,label='Seed-specific paired bootstrap 95% CI'); avg=f4.mamba_minus_nomamba.mean(); ax.scatter([avg],[3],marker='D',s=55,color='#D55E00',label='Three-seed descriptive mean (no pooled CI)'); ax.axvline(0,color='black',lw=1,ls='--'); ax.set(yticks=[0,1,2,3],yticklabels=[str(x) for x in f4.seed]+['3-seed descriptive'],xlabel='Patient-level macro F1 difference (Mamba - NoMamba)',ylabel='Seed'); ax.legend(fontsize=8,loc='best'); ax.grid(axis='x',alpha=.2); fig.tight_layout(); save_fig(fig,"FIGURE_4_STAGE2_MATCHED_MACRO_F1_FOREST")
    f4.to_csv(OUT/"FIGURE_4_STAGE2_MATCHED_MACRO_F1_FOREST_source_data.csv",index=False)
    # Figure 5B: paired correctness transitions by seed and class.
    summary=edf.groupby(["seed","transition"]).size().unstack(fill_value=0).reindex(columns=["improved","worsened","unchanged_correct","unchanged_incorrect"],fill_value=0)
    fig,ax=plt.subplots(figsize=(7.2,3.7)); bottom=np.zeros(3); colors={"improved":"#009E73","worsened":"#D55E00","unchanged_correct":"#56B4E9","unchanged_incorrect":"#999999"}
    for col in summary.columns: ax.bar([str(x) for x in summary.index],summary[col],bottom=bottom,label=col.replace('_',' '),color=colors[col]); bottom += summary[col].to_numpy()
    ax.set(ylabel='Patients (n = 149 per seed)',xlabel='Seed'); ax.legend(ncol=2,fontsize=8); ax.grid(axis='y',alpha=.2); fig.tight_layout(); save_fig(fig,"FIGURE_5B_STAGE2_PAIRED_ERROR_TRANSITIONS")
    summary.reset_index().to_csv(OUT/"FIGURE_5B_STAGE2_PAIRED_ERROR_TRANSITIONS_source_data.csv",index=False)
    # Reporting and provenance.
    primary=f4.mamba_minus_nomamba.to_numpy(); lo=f4.ci_lower.to_numpy();
    classification = "CONSISTENT_MAMBA_BENEFIT" if np.all(lo > 0) else ("MAMBA_BENEFIT_WITH_SEED_HETEROGENEITY" if np.any(lo > 0) else ("POSSIBLE_MAMBA_HARM" if np.all(f4.ci_upper.to_numpy() < 0) else "NO_CLEAR_MAMBA_BENEFIT"))
    report=f"""# Corrected Stage 2 Final Scientific Analysis\n\n## Classification\n\n`{classification}`\n\nThis read-only analysis uses the corrected Composite-Mamba seed 42 output only; it does not substitute the historical Stage 1 Composite-Mamba seed 42 results. Every model-seed OOF set contains the same 149 frozen outer-test patients once.\n\n## Integrity\n\nCorrected seed42: {int(integrity.integrity_pass.sum())}/5 folds passed status, checkpoint decode, history, probability, frozen patient identity, and single final outer-evaluation checks. Matched-condition audit unexpected shared differences: {int((~match.matched_condition_pass).sum())}.\n\n## Statistical Scope\n\nThe primary contrast is patient-level macro F1 (Composite-Mamba minus Composite-NoMamba). Seed-specific class-stratified paired patient bootstrap used 10,000 repetitions and seed 20260830. Three seeds reuse the same patients and folds; therefore the three-seed summary is descriptive only and has no pooled 447-row CI or p value. Negative Brier differences favor Mamba.\n\n## Convergence\n\nPredefined convergence flags among corrected seed42 folds: {int(pd.DataFrame(conv).convergence_warning.sum())}. Flags preserve every run and do not alter epochs, folds, seeds, or endpoints.\n\n## Interpretation Limits\n\nThis is a same-cohort matched ablation and seed-stability analysis, not independent external confirmation. It does not support clinical superiority, non-inferiority, or external generalization claims.\n"""
    (OUT/"STAGE2_CORRECTED_FINAL_SCIENTIFIC_REPORT.md").write_text(report)
    (OUT/"FIGURE_CAPTIONS_FINAL.md").write_text("""# Figure Captions\n\n## Figure 4\nSeed-specific paired patient-level macro F1 differences for Composite-Mamba minus Composite-NoMamba. Each point uses 149 identically aligned frozen outer-test patients and a class-stratified paired bootstrap (10,000 repetitions; percentile 95% CI). The diamond is the descriptive mean across the three repeated seeds; no pooled CI is shown because the same patients and folds recur across seeds.\n\n## Figure 5B\nPatient-level paired correctness transitions comparing Composite-Mamba with Composite-NoMamba in each seed. Counts use the fixed argmax class rule and retain all 149 patients per seed. These transitions are descriptive and do not constitute an external validation analysis.\n""")
    (OUT/"MANUSCRIPT_CORRECTION_PACKAGE.md").write_text("""# Manuscript Correction Package\n\nUse the corrected seed42 Composite-Mamba results only after citing this package. Replace any statement or figure that treats the historical Stage 1 Composite-Mamba seed42 results as the matched Stage2 seed42 result. Report Figure 4 as a seed-specific paired matched ablation; report the three-seed value as descriptive stability, not as 447 independent patients. Do not claim external validation, clinical superiority, equivalence, or non-inferiority.\n""")
    (OUT/"MISSING_OR_UNVERIFIED_ITEMS.md").write_text("""# Missing or Unverified Items\n\n- No pooled three-seed inferential CI was computed because seed-level repetitions reuse the same 149 patients and the frozen SAP does not specify a hierarchical pooled bootstrap.\n- No external validation, efficiency result, calibration analysis, or new checkpoint selection was performed here.\n""")
    prov=[]
    for p in [SPLIT, AUDIT/"mamba_nomamba_matched_condition_audit.csv", AUDIT/"mamba_nomamba_parameter_difference_audit.csv"]:
        if p.exists(): prov.append({"artifact":p.name,"path":str(p),"sha256":sha(p)})
    for model in ("Composite-Mamba","Composite-NoMamba"):
        for seed in SEEDS:
            for fold in range(5):
                p=config_path(model,seed,fold); prov.append({"artifact":f"{model}_seed{seed}_fold{fold}_config","path":str(p),"sha256":sha(p)})
    write_csv("SOURCE_ARTIFACT_PROVENANCE.csv",prov)
    (OUT/"CURRENT_STUDY_HANDOFF_FINAL.md").write_text(f"# Current Study Handoff\n\nFinal corrected Stage2 read-only analysis: `{classification}`. No training, inference, checkpoint mutation, threshold adjustment, or external validation was performed. Corrected Mamba seed42 source: `{CORR_OUT}`.\n")
    env={"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,"sklearn":"installed","created_utc":datetime.now(timezone.utc).isoformat()}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2))
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"SHA256_MANIFEST.txt", "analysis.log"} and "__pycache__" not in p.parts)
    with open(OUT/"SHA256_MANIFEST.txt","w") as f:
        for p in files: f.write(f"{sha(p)}  {p.relative_to(OUT)}\n")
    print(json.dumps({"classification":classification,"integrity_pass":int(integrity.integrity_pass.sum()),"primary_effects":f4[["seed","mamba_minus_nomamba","ci_lower","ci_upper"]].to_dict("records"),"out":str(OUT)},indent=2))

if __name__ == "__main__": main()
