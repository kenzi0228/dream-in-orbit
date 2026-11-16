#!/usr/bin/env bash
# Compile rapport.tex en rapport.pdf (XeLaTeX).
#   - avec Tectonic (interface V1) : enchaîne automatiquement les passes XeLaTeX et BibTeX ; le binaire
#     est pris, dans l'ordre, dans la variable TECTONIC (chemin d'un binaire), sur le PATH, puis dans
#     ../tools/tectonic (copie locale, non versionnée, de la version officielle) ;
#   - sinon avec latexmk (-xelatex), qui fait de même ;
#   - sinon avec xelatex et bibtex appelés à la main (xelatex, bibtex, xelatex, xelatex) ;
#   - sinon, explique comment installer un moteur.
# Avant de compiler, le script vérifie que les cinq figures sont dans figures/ (produites par
# figures.py, depuis la racine du dépôt : .venv/bin/python rapport/figures.py).
# Après la compilation, réussie ou non, il affiche les avertissements du journal, puis nettoie les
# fichiers auxiliaires en conservant rapport.pdf, rapport.log (relecture des avertissements) et
# rapport.bbl (bibliographie compilée, exigée par arXiv et la plupart des revues).
# Usage : ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

for f in reve_vs_realite zoom_horaire derive erreur_par_circuit calibration_incertitude; do
  if [ ! -f "figures/$f.pdf" ]; then
    echo "figure manquante : figures/$f.pdf (lancer, depuis la racine du dépôt : .venv/bin/python rapport/figures.py)" >&2
    exit 1
  fi
done

nettoyer() {
  rm -f rapport.aux rapport.blg rapport.out rapport.toc \
        rapport.fls rapport.fdb_latexmk rapport.xdv rapport.run.xml rapport.synctex.gz
}

resume_journal() {
  if [ -f rapport.log ]; then
    echo "--- rapport.log : Overfull / Underfull / Warning / undefined ---"
    grep -E 'Overfull|Underfull|Warning|undefined' rapport.log || echo "(aucun avertissement)"
    echo "--- fin du résumé ---"
  fi
}

# Échec du moteur : le résumé du journal est affiché (c'est là qu'il sert le plus), les fichiers
# auxiliaires sont nettoyés (rapport.log est conservé) et le script rend un code de sortie non nul.
echec() {
  echo "La compilation a échoué (journal : $(pwd)/rapport.log)." >&2
  resume_journal
  nettoyer
  exit 1
}

TECTONIC_BIN=""
if [ -n "${TECTONIC:-}" ] && [ -x "${TECTONIC}" ]; then
  TECTONIC_BIN="${TECTONIC}"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC_BIN="$(command -v tectonic)"
elif [ -x ../tools/tectonic ]; then
  TECTONIC_BIN="$(cd ../tools && pwd)/tectonic"
fi

if [ -n "$TECTONIC_BIN" ]; then
  echo "Moteur : $TECTONIC_BIN ($("$TECTONIC_BIN" --version 2>/dev/null | head -n 1))"
  "$TECTONIC_BIN" --keep-logs --keep-intermediates rapport.tex || echec
elif command -v latexmk >/dev/null 2>&1; then
  echo "Moteur : latexmk -xelatex"
  latexmk -xelatex -interaction=nonstopmode -halt-on-error rapport.tex || echec
elif command -v xelatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  echo "Moteur : xelatex + bibtex (xelatex, bibtex, xelatex, xelatex)"
  xelatex -interaction=nonstopmode -halt-on-error rapport.tex || echec
  bibtex rapport || echec
  xelatex -interaction=nonstopmode -halt-on-error rapport.tex || echec
  xelatex -interaction=nonstopmode -halt-on-error rapport.tex || echec
else
  cat >&2 <<'MSG'
Aucun moteur LaTeX trouvé (ni TECTONIC, ni « tectonic » sur le PATH, ni ../tools/tectonic, ni « latexmk »,
ni « xelatex » accompagné de « bibtex »).
Deux façons d'en installer un :
  - Tectonic (léger, télécharge les paquets à la demande) :
      macOS : brew install tectonic        Linux/Windows : https://tectonic-typesetting.github.io/
  - TeX Live complet (fournit xelatex, bibtex, latexmk, les polices TeX Gyre et Latin Modern) :
      https://tug.org/texlive/   (macOS : MacTeX, https://tug.org/mactex/)
Puis relancer ./build.sh dans ce dossier.
MSG
  exit 1
fi

resume_journal
nettoyer
echo "Terminé : $(pwd)/rapport.pdf (journal : rapport.log ; bibliographie compilée : rapport.bbl)"
