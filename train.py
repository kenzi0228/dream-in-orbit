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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tf-epochs", type=int, default=15, help="epoques en teacher forcing pur avant le scheduled sampling")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--windows", type=int, default=256, help="fenetres tirees par epoque")
    ap.add_argument("--dream", type=int, default=24 * 14, help="longueur du reve dans chaque fenetre (heures)")
    ap.add_argument("--noise", type=float, default=0.05, help="bruit sur l'etat re-injecte (unites standardisees)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--drop-groups", default="", help="groupes de features a retirer (saaf,ltdata,evtf,ftl,dmop_base,dmop_switch,dmop_cmd ; dmop = les trois)")
    ap.add_argument("--keep-groups", default="saaf,ltdata,evtf,ftl", help="groupes de features a garder (vide = tous)")
    ap.add_argument("--drop-cols", default="lt_earthmars|lt_sunmarsearth",
                    help="regex : colonnes de features a retirer (par defaut les 'calendriers' : distance Terre-Mars, angle Soleil-Mars-Terre)")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--feat-dropout", type=float, default=0.2)
    ap.add_argument("--input-noise", type=float, default=0.0, help="bruit gaussien sur les features standardisees (train)")
    ap.add_argument("--init-bias", type=float, default=0.0, help="correction de niveau a partir des 7 jours (validation)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--stateful", action="store_true", default=True,
                    help="entrainement par flux continus : l'etat cache est transmis entre segments (BPTT tronquee), comme en simulation longue")
    ap.add_argument("--windows-mode", dest="stateful", action="store_false", help="ablation : fenetres aleatoires independantes (etat remis a zero)")
    ap.add_argument("--streams", type=int, default=16, help="stateful : nombre de flux paralleles (decoupage du train)")
    ap.add_argument("--seg", type=int, default=168, help="stateful : longueur des segments de BPTT (heures)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def masked_mse(pred_s, y_s, mask, y_std):
    """MSE sur l'echelle brute (celle de la metrique) : (pred - y)^2 * std^2, en ignorant les heures sans verite."""
    err = ((pred_s - y_s) * y_std) ** 2
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
                feedback=not args.no_feedback, dropout=args.dropout, feat_dropout=args.feat_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stateful:
        n_batches = ((len(xs) - args.seg) // args.streams) // args.seg
    else:
        n_batches = max(1, args.windows // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches + 1, pct_start=0.15)
    L = N_INIT + args.dream
    T = len(xs)
    history = []
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
                loss = masked_mse(pred, yb, mb, y_std)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
                h, yp = state
                state = (None if h is None else h.detach(), yp.detach())
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
            loss = masked_mse(pred, yb, mb, y_std)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"seed": seed, "epoch": ep + 1, "p_tf": round(p_tf, 3), "train_rmse": math.sqrt(tot / n_batches), "time_s": round(time.time() - t0, 1)}
        if Xva is not None and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            t1 = time.time()
            pred = simulate_one(model, norm, Xva, Yva, N_INIT, init_bias=args.init_bias)
            rec["val_rmse"] = rmse(pred[N_INIT:], Yva[N_INIT:])
            rec["val_time_s"] = round(time.time() - t1, 1)
        history.append(rec)
        log(json.dumps(rec))
    ck = {"state_dict": model.state_dict(), "normalizer": norm.state_dict(),
          "config": {"hidden": args.hidden, "layers": args.layers, "feedback": not args.no_feedback,
                     "dropout": args.dropout, "feat_dropout": args.feat_dropout, "no_memory": args.no_memory, "seed": seed}}
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
    (out_dir / "meta.json").write_text(json.dumps(meta))

    preds = []
    for s in range(args.seeds):
        model, norm, hist = train_one(s, Xtr, Ytr, Xva, Yva, feats, targets, args, out_dir, log)
        if Xva is not None:
            preds.append(simulate_one(model, norm, Xva, Yva, N_INIT, init_bias=args.init_bias))
            ens = np.mean(preds, axis=0)
            log(json.dumps({"ensemble_size": len(preds), "val_rmse_ensemble": rmse(ens[N_INIT:], Yva[N_INIT:])}))
    if Xva is not None:
        ens = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        report = {
            "val_rmse": rmse(ens[N_INIT:], Yva[N_INIT:]),
            "val_rmse_per_circuit": dict(zip(targets, rmse_per_circuit(ens[N_INIT:], Yva[N_INIT:]).round(5).tolist())),
            "val_rmse_by_week": rmse_by_horizon(ens[N_INIT:], Yva[N_INIT:]),
        }
        (out_dir / "val_report.json").write_text(json.dumps(report, indent=1))
        pd.DataFrame(ens, index=val_index, columns=targets).to_csv(out_dir / "val_pred.csv")
        pd.DataFrame(std, index=val_index, columns=targets).to_csv(out_dir / "val_std.csv")
        log(json.dumps({"final_val_rmse": report["val_rmse"]}))
    log(json.dumps({"done": str(out_dir)}))


if __name__ == "__main__":
    main()
