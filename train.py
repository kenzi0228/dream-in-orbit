"""
Entraine le world model (ensemble de GRU) a partir du kit, en une commande.

    python train.py                       # final : annee1 + annee2 -> weights/
    python train.py --holdout             # annee1 -> train, annee2 -> validation, -> weights_holdout/

Options principales :
    --seeds 5            taille de l'ensemble (Deep Ensembles : moyenne = prediction, ecart-type = incertitude)
    --epochs 30          nombre d'epoques (les premieres en teacher forcing, puis scheduled sampling)
    --no-feedback        ablation : le GRU ne relit pas l'etat precedent (planning seul)
    --drop-groups a,b    ablation : retire des groupes de features (saaf, ltdata, evtf, ftl, dmop)
    --no-memory          ablation : MLP heure par heure (pas de GRU)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dio.data import get_vocab, load_period, select_features
from dio.features import H
from dio.gbm import GBMComponent
from dio.metrics import rmse, rmse_by_horizon, rmse_per_circuit
from dio.model import MLPModel, Normalizer, WorldModel
from dio.simulate import simulate_one

N_INIT = 24 * 7  # 7 jours reels


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--holdout", action="store_true", help="train annee1, validation annee2 (protocole strict)")
    ap.add_argument("--holdout-split", type=float, default=0.0,
                    help="validation chronologique : train = annee1 + debut de annee2 (fraction F), validation = fin de annee2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0, help="premiere graine (les graines vont de offset a offset + seeds - 1)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tf-epochs", type=int, default=15, help="epoques en teacher forcing pur avant le scheduled sampling")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--cell", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--swa", type=int, default=0, help="moyenne des poids des K dernieres epoques (0 = desactive)")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--windows", type=int, default=256, help="fenetres tirees par epoque")
    ap.add_argument("--dream", type=int, default=24 * 14, help="longueur du reve dans chaque fenetre (heures)")
    ap.add_argument("--noise", type=float, default=0.05, help="bruit sur l'etat re-injecte (unites standardisees)")
    ap.add_argument("--huber", type=float, default=0.0, help="seuil de la perte de Huber sur l'echelle brute (0 = MSE)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--drop-groups", default="", help="groupes de features a retirer (saaf,ltdata,evtf,ftl,dmop_base,dmop_switch,dmop_cmd ; dmop = les trois)")
    ap.add_argument("--keep-groups", default="saaf,ltdata,evtf,ftl,dmop_stem_cnt",
                    help="groupes de features du GRU (vide = tous) ; dmop_stem_cnt = comptes horaires de commandes par stem, sans la lettre de variante")
    ap.add_argument("--drop-cols", default="lt_earthmars|lt_sunmarsearth",
                    help="regex : colonnes de features a retirer (par defaut les 'calendriers' : distance Terre-Mars, angle Soleil-Mars-Terre)")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--feat-dropout", type=float, default=0.2)
    ap.add_argument("--input-noise", type=float, default=0.0, help="bruit gaussien sur les features standardisees (train)")
    ap.add_argument("--init-bias", type=float, default=0.0, help="correction de niveau a partir des 7 jours (validation)")
    ap.add_argument("--adapt-steps", type=int, default=0, help="adaptation au test : pas de gradient sur les 7 jours reels avant le reve")
    ap.add_argument("--adapt-lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--stateful", action="store_true", default=True,
                    help="entrainement par flux continus : l'etat cache est transmis entre segments (BPTT tronquee), comme en simulation longue")
    ap.add_argument("--windows-mode", dest="stateful", action="store_false", help="ablation : fenetres aleatoires independantes (etat remis a zero)")
    ap.add_argument("--streams", type=int, default=16, help="stateful : nombre de flux paralleles (decoupage du train)")
    ap.add_argument("--seg", type=int, default=168, help="stateful : longueur des segments de BPTT (heures)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gbm-weight", type=float, default=0.5,
                    help="poids du composant gradient boosting dans le melange final (0 = ensemble de GRU pur)")
    ap.add_argument("--gbm-groups", default="", help="groupes de features du GBM (vide = toutes les colonnes)")
    ap.add_argument("--gbm-bags", type=int, default=5, help="bagging temporel du GBM : nombre de sacs (0 ou 1 = un seul ajustement)")
    ap.add_argument("--gbm-drop-cols", default="lt_earthmars|lt_sunmarsearth|^roll_|^ewm_|^dm_s_|^dm_c_|^dm_r_",
                    help="regex : colonnes retirees des features du GBM (par defaut les 'calendriers')")
    ap.add_argument("--aux-models", type=int, default=0,
                    help="GRU auxiliaires entraines sur un autre jeu de features (diversite de l'ensemble, reseau de neurones pur)")
    ap.add_argument("--aux-groups", default="", help="groupes de features des GRU auxiliaires (vide = toutes)")
    ap.add_argument("--aux-weight", type=float, default=0.2, help="poids total des GRU auxiliaires dans la moyenne")
    ap.add_argument("--stack", action="store_true",
                    help="empilement : le GBM predit d'abord, le GRU apprend le residu (cible = y - GBM hors-pli) ; "
                         "prediction finale = GBM + GRU. Remplace le melange pondere.")
    ap.add_argument("--stack-folds", type=int, default=4, help="empilement : nombre de blocs temporels pour les predictions hors-pli")
    ap.add_argument("--stack-input", action="store_true", help="empilement : la prediction du GBM est aussi donnee en entree du GRU")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def masked_mse(pred_s, y_s, mask, y_std, huber: float = 0.0):
    """MSE sur l'echelle brute (celle de la metrique) : (pred - y)^2 * std^2, en ignorant les heures sans verite.
    huber > 0 : perte de Huber (quadratique sous le seuil, lineaire au-dela) sur l'echelle brute, plus robuste
    aux heures aberrantes (changements de regime)."""
    d = (pred_s - y_s) * y_std
    if huber > 0:
        a = d.abs()
        err = torch.where(a <= huber, 0.5 * d ** 2, huber * (a - 0.5 * huber)) * 2  # x2 : meme echelle que la MSE sous le seuil
    else:
        err = d ** 2
    err = err * mask.unsqueeze(-1)
    return err.sum() / (mask.sum() * y_s.shape[-1] + 1e-9)


def sample_windows(T, L, n, rng):
    return rng.integers(0, T - L, size=n)


def train_one(seed, Xtr, Ytr, Xva, Yva, feats, targets, args, out_dir, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Normalizer.fit(Xtr, np.nan_to_num(Ytr, nan=np.nanmean(Ytr, axis=0)))
    xs = norm.x(Xtr)
    Yf = pd.DataFrame(Ytr).ffill().bfill().to_numpy()
    ys = norm.y(Yf)
    mask_all = torch.as_tensor(~np.isnan(Ytr).any(axis=1), dtype=torch.float32)
    y_std = norm.y_std

    cls = MLPModel if args.no_memory else WorldModel
    model = cls(n_feat=len(feats), n_out=len(targets), hidden=args.hidden, layers=args.layers,
                feedback=not args.no_feedback, dropout=args.dropout, feat_dropout=args.feat_dropout, cell=args.cell)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stateful:
        n_batches = ((len(xs) - args.seg) // args.streams) // args.seg
    else:
        n_batches = max(1, args.windows // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches + 1, pct_start=0.15)
    L = N_INIT + args.dream
    T = len(xs)
    history = []
    swa_sum, swa_n = None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        # scheduled sampling : p_tf = 1 pendant les tf_epochs, puis decroit lineairement vers 0
        if ep < args.tf_epochs:
            p_tf = 1.0
        else:
            p_tf = max(0.0, 1.0 - (ep - args.tf_epochs + 1) / max(1, args.epochs - args.tf_epochs - 2))
        tot = 0.0
        if args.stateful:
            # B flux contigus (decalage aleatoire a chaque epoque), segments successifs, etat transmis
            B = args.streams
            Ls = (T - args.seg) // B
            off = int(rng.integers(0, max(1, T - Ls * B)))
            starts = off + np.arange(B) * Ls
            state = None
            n_seg = Ls // args.seg
            for k in range(n_seg):
                idx = starts + k * args.seg
                xb = torch.stack([xs[i:i + args.seg] for i in idx])
                yb = torch.stack([ys[i:i + args.seg] for i in idx])
                mb = torch.stack([mask_all[i:i + args.seg] for i in idx])
                if args.input_noise > 0:
                    xb = xb + args.input_noise * torch.randn_like(xb)
                n_init = N_INIT if k == 0 else 0
                pred, state = model.rollout(xb, yb, n_init=n_init, p_tf=p_tf, noise=args.noise, state=state, return_state=True)
                loss = masked_mse(pred, yb, mb, y_std, args.huber)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
                h, yp = state
                if isinstance(h, tuple):
                    h = tuple(t.detach() for t in h)
                elif h is not None:
                    h = h.detach()
                state = (h, yp.detach())
            n_batches = n_seg
        for b in range(0 if args.stateful else n_batches):
            idx = sample_windows(T, L, args.batch, rng)
            xb = torch.stack([xs[i:i + L] for i in idx])
            yb = torch.stack([ys[i:i + L] for i in idx])
            mb = torch.stack([mask_all[i:i + L] for i in idx])
            if args.input_noise > 0:
                xb = xb + args.input_noise * torch.randn_like(xb)
            if p_tf >= 1.0:
                pred = model.forward_tf(xb, yb)
            else:
                pred = model.rollout(xb, yb, n_init=N_INIT, p_tf=p_tf, noise=args.noise)
            loss = masked_mse(pred, yb, mb, y_std, args.huber)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if args.swa and ep >= args.epochs - args.swa:
            sd = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            swa_sum = sd if swa_sum is None else {k: swa_sum[k] + sd[k] for k in sd}
            swa_n += 1
            if ep + 1 == args.epochs:
                model.load_state_dict({k: v / swa_n for k, v in swa_sum.items()})
        rec = {"seed": seed, "epoch": ep + 1, "p_tf": round(p_tf, 3), "train_rmse": math.sqrt(tot / n_batches), "time_s": round(time.time() - t0, 1)}
        if Xva is not None and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            t1 = time.time()
            pred = simulate_one(model, norm, Xva, Yva, N_INIT, init_bias=args.init_bias,
                                adapt_steps=args.adapt_steps, adapt_lr=args.adapt_lr)
            rec["val_rmse"] = rmse(pred[N_INIT:], Yva[N_INIT:])
            rec["val_time_s"] = round(time.time() - t1, 1)
        history.append(rec)
        log(json.dumps(rec))
    ck = {"state_dict": model.state_dict(), "normalizer": norm.state_dict(), "features": list(feats),
          "config": {"hidden": args.hidden, "layers": args.layers, "feedback": not args.no_feedback,
                     "dropout": args.dropout, "feat_dropout": args.feat_dropout, "no_memory": args.no_memory,
                     "cell": args.cell, "seed": seed}}
    torch.save(ck, out_dir / f"model_{seed}.pt")
    return model, norm, history


def main():
    args = parse()
    if args.threads:
        torch.set_num_threads(args.threads)
    kit = Path(args.kit)
    out_dir = Path(args.out or ("weights_holdout" if (args.holdout or args.holdout_split > 0) else "weights"))
    if args.tag:
        out_dir = Path(str(out_dir) + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "train_log.jsonl", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    vocab = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", vocab)
    X2, Y2 = load_period(kit / "annee2", vocab)
    feats = select_features(X1.columns, keep=args.keep_groups, drop=args.drop_groups)
    if args.drop_cols:
        import re
        feats = [c for c in feats if not re.search(args.drop_cols, c)]
    targets = list(Y1.columns)
    if args.holdout:
        Xtr, Ytr = X1[feats].to_numpy(), Y1.to_numpy()
        Xva, Yva = X2[feats].to_numpy(), Y2.to_numpy()
        val_index = X2.index
    elif args.holdout_split > 0:
        k = int(len(X2) * args.holdout_split)
        Xtr = pd.concat([X1, X2.iloc[:k]])[feats].to_numpy()
        Ytr = pd.concat([Y1, Y2.iloc[:k]]).to_numpy()
        Xva, Yva = X2.iloc[k:][feats].to_numpy(), Y2.iloc[k:].to_numpy()
        val_index = X2.index[k:]
    else:
        Xtr = pd.concat([X1, X2])[feats].to_numpy()
        Ytr = pd.concat([Y1, Y2]).to_numpy()
        Xva = Yva = None
    log(json.dumps({"train_hours": len(Xtr), "val_hours": 0 if Xva is None else len(Xva), "n_features": len(feats), "args": vars(args)}))
    meta = {"features": feats, "targets": targets, "vocab": vocab, "n_init": N_INIT, "args": vars(args)}
    (out_dir / "meta.json").write_text(json.dumps(meta))  # ecrit tot (poids partiels utilisables), complete a la fin

    # ---- empilement : GBM hors-pli sur le train, le GRU apprend le residu
    gbm_feats_all = None
    Ytr_target, Yva_target, pg_va = Ytr, Yva, None
    if args.stack:
        import re as _re
        gbm_feats_all = select_features(X1.columns, keep=args.gbm_groups, drop="")
        if args.gbm_drop_cols:
            gbm_feats_all = [c for c in gbm_feats_all if not _re.search(args.gbm_drop_cols, c)]
        Xg_all = (pd.concat([X1, X2]) if Xva is None else (X1 if args.holdout else pd.concat([X1, X2.iloc[:int(len(X2) * args.holdout_split)]])))[gbm_feats_all].to_numpy()
        t0 = time.time()
        oof = np.full(Ytr.shape, np.nan, dtype=np.float64)
        bounds = np.linspace(0, len(Xg_all), args.stack_folds + 1).astype(int)
        for i in range(args.stack_folds):
            a, b = bounds[i], bounds[i + 1]
            tr = np.r_[0:a, b:len(Xg_all)]
            g = GBMComponent(gbm_feats_all).fit(Xg_all[tr], Ytr[tr])
            oof[a:b] = g.predict(Xg_all[a:b])
        log(json.dumps({"stack_oof_done": args.stack_folds, "oof_rmse": rmse(oof[~np.isnan(Ytr).any(axis=1)], Ytr[~np.isnan(Ytr).any(axis=1)]), "time_s": round(time.time() - t0)}))
        Ytr_target = Ytr - oof  # residu (NaN la ou y est NaN)
        if Xva is not None:
            g_full = GBMComponent(gbm_feats_all, bags=args.gbm_bags).fit(Xg_all, Ytr)
            Xg_va = (X2 if args.holdout else X2.iloc[int(len(X2) * args.holdout_split):])[gbm_feats_all].to_numpy()
            pg_va = g_full.predict(Xg_va)
            Yva_target = Yva - pg_va
            log(json.dumps({"val_rmse_gbm_alone": rmse(pg_va[N_INIT:], Yva[N_INIT:])}))
        if args.stack_input:
            gcols = [f"gbm_{t}" for t in targets]
            Xtr = np.concatenate([Xtr, oof], axis=1)
            feats = feats + gcols
            if Xva is not None:
                Xva = np.concatenate([Xva, pg_va], axis=1)
            meta["features"] = feats
        meta["stack"] = True

    preds, weights = [], []
    model_weights = {}
    for s in range(args.seed_offset, args.seed_offset + args.seeds):
        model, norm, hist = train_one(s, Xtr, Ytr_target, Xva, Yva_target, feats, targets, args, out_dir, log)
        model_weights[f"model_{s}.pt"] = 1.0
        if Xva is not None:
            pr = simulate_one(model, norm, Xva, Yva_target, N_INIT, init_bias=args.init_bias,
                              adapt_steps=args.adapt_steps, adapt_lr=args.adapt_lr, clip=not args.stack)
            preds.append(pr if not args.stack else np.clip(pr + pg_va, 0, None))
            weights.append(1.0)
            ens = np.average(preds, axis=0, weights=weights)
            log(json.dumps({"ensemble_size": len(preds), "val_rmse_ensemble": rmse(ens[N_INIT:], Yva[N_INIT:])}))

    # GRU auxiliaires (autre jeu de features), poids total aux_weight
    if args.aux_models > 0:
        aux_feats = select_features(X1.columns, keep=args.aux_groups, drop="")
        aux_tr = (pd.concat([X1, X2]) if Xva is None else (X1 if args.holdout else pd.concat([X1, X2.iloc[:int(len(X2) * args.holdout_split)]])))[aux_feats].to_numpy()
        aux_va = None if Xva is None else (X2 if args.holdout else X2.iloc[int(len(X2) * args.holdout_split):])[aux_feats].to_numpy()
        w_main = (1 - args.aux_weight) / args.seeds
        w_aux = args.aux_weight / args.aux_models
        model_weights = {k: w_main for k in model_weights}
        weights = [w_main] * len(weights)
        for s in range(100, 100 + args.aux_models):
            model, norm, hist = train_one(s, aux_tr, Ytr, aux_va, Yva, aux_feats, targets, args, out_dir, log)
            model_weights[f"model_{s}.pt"] = w_aux
            if Xva is not None:
                preds.append(simulate_one(model, norm, aux_va, Yva, N_INIT))
                weights.append(w_aux)
                ens = np.average(preds, axis=0, weights=weights)
                log(json.dumps({"ensemble_size": len(preds), "val_rmse_ensemble": rmse(ens[N_INIT:], Yva[N_INIT:])}))
        meta["aux_features"] = aux_feats
    meta["model_weights"] = model_weights

    # composant gradient boosting (en mode empilement : base de la prediction, le GRU ajoute le residu)
    gbm = None
    if args.stack:
        args.gbm_weight = 1.0
    if args.gbm_weight > 0:
        import re as _re
        gbm_feats = select_features(X1.columns, keep=args.gbm_groups, drop="")
        if args.gbm_drop_cols:
            gbm_feats = [c for c in gbm_feats if not _re.search(args.gbm_drop_cols, c)]
        Xg_tr = (pd.concat([X1, X2]) if Xva is None else (X1 if args.holdout else pd.concat([X1, X2.iloc[:int(len(X2) * args.holdout_split)]])))[gbm_feats].to_numpy()
        t0 = time.time()
        gbm = GBMComponent(gbm_feats, bags=args.gbm_bags).fit(Xg_tr, Ytr)
        gbm.save(out_dir / "gbm.pkl")
        meta["gbm_weight"] = args.gbm_weight
        meta["gbm_features_n"] = len(gbm_feats)
        log(json.dumps({"gbm_trained": len(gbm_feats), "time_s": round(time.time() - t0)}))
    (out_dir / "meta.json").write_text(json.dumps(meta))

    if Xva is not None:
        ens = np.average(preds, axis=0, weights=weights)
        std = np.sqrt(np.average((np.stack(preds) - ens) ** 2, axis=0, weights=weights))
        report = {"val_rmse_gru": rmse(ens[N_INIT:], Yva[N_INIT:])}
        if gbm is not None and not args.stack:
            Xg_va = (X2 if args.holdout else X2.iloc[int(len(X2) * args.holdout_split):])[gbm_feats].to_numpy()
            pg = gbm.predict(Xg_va)
            report["val_rmse_gbm"] = rmse(pg[N_INIT:], Yva[N_INIT:])
            for w in [0.2, 0.3, 0.4]:
                report[f"val_rmse_blend_{w}"] = rmse(((1 - w) * ens + w * pg)[N_INIT:], Yva[N_INIT:])
            ens = (1 - args.gbm_weight) * ens + args.gbm_weight * pg
        elif args.stack:
            report["val_rmse_gbm"] = rmse(pg_va[N_INIT:], Yva[N_INIT:])  # ens contient deja GBM + residu
        report.update({
            "val_rmse": rmse(ens[N_INIT:], Yva[N_INIT:]),
            "val_rmse_per_circuit": dict(zip(targets, rmse_per_circuit(ens[N_INIT:], Yva[N_INIT:]).round(5).tolist())),
            "val_rmse_by_week": rmse_by_horizon(ens[N_INIT:], Yva[N_INIT:]),
        })
        (out_dir / "val_report.json").write_text(json.dumps(report, indent=1))
        pd.DataFrame(ens, index=val_index, columns=targets).to_csv(out_dir / "val_pred.csv")
        pd.DataFrame(std, index=val_index, columns=targets).to_csv(out_dir / "val_std.csv")
        log(json.dumps({"final_val_rmse": report["val_rmse"]}))
    log(json.dumps({"done": str(out_dir)}))


if __name__ == "__main__":
    main()
