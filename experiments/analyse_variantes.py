"""Derive du vocabulaire DMOP entre annees : lettres de variantes par stem, commandes presentes dans une seule annee."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.features import split_stem
H = 3600000
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
d = {n: pd.read_csv(kit / n / "dmop.csv") for n in ["annee1", "annee2"]}
for n in d:
    d[n]["stem"], d[n]["var"] = zip(*[split_stem(c) for c in d[n]["subsystem"]])
c1 = d["annee1"]["subsystem"].value_counts(); c2 = d["annee2"]["subsystem"].value_counts()
cmds = v["cmds"]
only1 = [c for c in cmds if c2.get(c, 0) < 5]; only2 = [c for c in cmds if c1.get(c, 0) < 5]
print(f"vocabulaire : {len(cmds)} commandes ; presentes quasi uniquement en annee1 : {len(only1)} ; uniquement en annee2 : {len(only2)}")
print("  ex. annee1 seules :", only1[:15]); print("  ex. annee2 seules :", only2[:15])
# part des emissions de commandes (annee2) dont la commande exacte est dans le vocabulaire / dont le stem est dans le vocabulaire des stems
stems = set(v.get("stems", []))
for n in d:
    s = d[n]; print(f"  {n}: emissions couvertes par le vocab commandes : {100*s['subsystem'].isin(set(cmds)).mean():.1f}% ; par le vocab stems : {100*s['stem'].isin(stems).mean():.1f}%")
# interrupteurs : variantes par annee
sw = {}
for st, va in v.get("switches", []): sw.setdefault(st, []).append(va)
print(f"\n{len(sw)} stems d'interrupteurs ; variantes (>=5 emissions) par annee :")
rows = []
for st, vs in sw.items():
    v1 = d["annee1"][d["annee1"].stem == st]["var"].value_counts(); v2 = d["annee2"][d["annee2"].stem == st]["var"].value_counts()
    s1 = set(v1[v1 >= 5].index); s2 = set(v2[v2 >= 5].index)
    rows.append((st, "".join(sorted(s1)), "".join(sorted(s2)), len(s1 & s2), len(s1 ^ s2), int(v1.sum()), int(v2.sum())))
df = pd.DataFrame(rows, columns=["stem", "var_a1", "var_a2", "communes", "differentes", "n_a1", "n_a2"])
print(df.sort_values("differentes", ascending=False).to_string(index=False))
print("\nstems dont le jeu de variantes change :", int((df.differentes > 0).sum()), "/", len(df), "; sans variante commune :", int((df.communes == 0).sum()))
# composition FTL pendant la conjonction annee2 (semaines 63-70) vs annee1 semaines 60-61 et 74-75
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
ftl = [c for c in X1.columns if c.startswith("ftl_")]
def wk(X, a, b): return X.iloc[(a-1)*168:b*168][ftl + ["ev_umbra", "lt_sunmars_km" if "lt_sunmars_km" in X else ftl[0]]].mean().round(3)
print("\nFTL annee2 sem 64-70 :"); print(wk(X2, 64, 70).to_string())
print("\nFTL annee1 sem 60-61 :"); print(wk(X1, 60, 61).to_string())
print("\nFTL annee1 sem 74-75 :"); print(wk(X1, 74, 75).to_string())
print("\nFTL annee2 sem 52-58 :"); print(wk(X2, 52, 58).to_string())
