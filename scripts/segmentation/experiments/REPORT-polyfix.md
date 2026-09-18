# Polygon fixer report

Caveat first: the ground truth polygons are corrected kraken polygons, so every
IoU column below is biased towards blla (kraken-like shapes score higher without
being better). Neighbour defects, height over pitch, degenerate counts and ink
loss do not use the ground truth shape and are unbiased.

## Coptic (10 pages, overall)

| variant | polygons | mean IoU | IoU>=0.5 | IoU>=0.7 | defects | height/pitch | degen |
|---|---|---|---|---|---|---|---|
| blla (v0-same10) | 1167 | 0.8588 | 0.9120 | 0.8157 | 39 | 1.429 | 0 |
| Orli orig (orli-fixed-poly) | 860/874 | 0.7717 | 0.9670 | 0.8791 | 381 | 1.710 | 14 |
| polyfix split 0.35 | 874 | 0.7265 | 0.9849 | 0.6731 | 0 | 1.303 | 0 |
| polyfix split 0.50 | 874 | 0.6220 | 0.9299 | 0.1813 | 0 | 1.217 | 0 |
| polyfix split 0.65 | 874 | 0.5094 | 0.5179 | 0.0302 | 0 | 1.087 | 0 |

## Greek (5 pages, view all, overall)

| variant | polygons | mean IoU | IoU>=0.5 | IoU>=0.7 | defects | height/pitch | degen |
|---|---|---|---|---|---|---|---|
| blla (greek-v0) | 334 | 0.8566 | 0.9956 | 0.9031 | 39 | 1.339 | 0 |
| Orli orig (greek-orli-folios) | 274 | 0.6975 | 0.9730 | 0.4595 | 39 | 1.499 | 0 |
| polyfix split 0.35 | 274 | 0.7291 | 0.9775 | 0.6486 | 0 | 1.149 | 0 |
| polyfix split 0.50 | 274 | 0.6976 | 0.9685 | 0.5225 | 0 | 1.074 | 0 |
| polyfix split 0.65 | 274 | 0.6054 | 0.8739 | 0.1441 | 0 | 0.951 | 0 |

## Ink loss (share of dark pixels in original union kept by fixed union, Otsu)

| corpus | split 0.35 | split 0.50 | split 0.65 |
|---|---|---|---|
| Coptic | 0.9942 | 0.9323 | 0.7718 |
| Greek | 0.9930 | 0.9636 | 0.8879 |

Retention above 1.0 on vat-2v (1.077), c10 (1.033), c12 (1.003), c13 (1.001) at
split 0.35 is the repair step: rebuilt bands cover ink that broken polygons missed.
Per page detail is in the ink_loss.py output quoted in the log, worst pages at
0.35 are c14 (0.977) and vat-1r (0.981), both fine.

## Ablation (Coptic, split 0.50, overall)

| run | mean IoU | defects | height/pitch | degen |
|---|---|---|---|---|
| full fix | 0.6220 | 0 | 1.217 | 0 |
| --no-trim | 0.7867 | 406 | 1.659 | 0 |
| --no-cap | 0.6198 | 0 | 1.266 | 0 |
| --no-repair | 0.6120 | 0 | 1.201 | 0 |
| --no-neighbours | 0.7939 | 317 | 1.586 | 0 |

## What each step changed

- Neighbours: the gate for everything. Without it trim does nothing and cap fires
  on both sides of every line; 317 of 381 defects survive. Overlap test (>=50% of
  the shorter baseline) keeps the two columns apart correctly.
- Trim: the defect killer. 381 to 0 on Coptic, 39 to 0 on Greek. It also costs the
  most IoU because kraken shapes overlap generously and ours no longer do.
- Cap: small in aggregate (height 1.217 vs 1.266 without it) but it fixes the
  headline bug: first-line huge boxes (e.g. vat-1r line 0 reached y=75, now capped
  at baseline minus one pitch). Only fires on column ends.
- Repair: 14 broken Coptic polygons rebuilt as baseline bands (vat-2v: 4, c10: 3,
  c11/c13: 2 each, vat-1r/vat-1v/vat-3r region: rest). Small aggregate effect,
  required for zero degenerate.
- Clamp: hygiene only (integer rounding, image bounds, dedupe). No metric moves.

## Recommended default

--split-above 0.35, not 0.5. On both corpora it has the best IoU (0.7265 Coptic,
0.7291 Greek), keeps 99%+ of ink (0.9942 / 0.9930), still kills all defects, and
height over pitch (1.303 / 1.149) stays below blla (1.429 / 1.339). Reason: text
sits above the baseline, so the boundary between two lines belongs closer to the
upper baseline; 0.5 cuts ascenders (ink 0.93/0.96) and 0.65 cuts hard (0.77/0.89).

This is now the code default: `--split-above` defaults to 0.35. A reviewer's
per-line ink check on 872 corrected Coptic lines agreed, ink recall 0.939 at 0.35
against 0.851 at 0.50, with foreign ink 2.5% against 2.9%. The runs recorded above
predate the change, so every command below that must reproduce a 0.50 number now
passes `--split-above 0.5` explicitly.

## Lines the fixer made worse (pred line indices, before -> after IoU at 0.35)

By biased IoU, 406 of 728 matched Coptic lines drop by more than 0.05 (92 improve,
230 tie; mean drop of the worse set -0.121). Worst three per page:

- c10 (45 worse): l62 0.850->0.517, l61 0.810->0.524, l36 0.857->0.626
- c11 (93 worse): l126 0.728->0.450, l121 0.896->0.652, l54 0.896->0.670
- c12 (56 worse): l30 0.775->0.509, l2 0.914->0.709, l60 0.864->0.668
- c13 (30 worse): l39 0.816->0.659, l100 0.748->0.593, l65 0.856->0.710
- c14 (83 worse): l65 0.811->0.501, l109 0.894->0.657, l38 0.912->0.687
- vat-1r (21 worse): l35 0.856->0.727, l46 0.867->0.770, l48 0.855->0.759
- vat-1v (31 worse): l46 0.934->0.755, l61 0.830->0.657, l19 0.882->0.725
- vat-2r (12 worse): l16 0.900->0.768, l47 0.820->0.702, l43 0.861->0.750
- vat-2v (11 worse): l33 0.848->0.698, l11 0.834->0.708, l42 0.857->0.731
- vat-3r (24 worse): l30 0.936->0.787, l8 0.910->0.770, l31 0.807->0.680

By unbiased shape quality no line got worse: zero remaining overlap defects, zero
degenerate, and ink retention >= 0.977 on every page. The IoU drops are the price
of leaving the kraken shape convention (tighter, non-overlapping segments).

## What I did not manage

- IoU stays below blla by construction; matching the kraken watershed shape was
  never a goal and the GT bias rewards exactly that shape.
- Local pitch falls back to a page median (30px if nothing found); isolated lines
  and single-line columns get generic caps.
- Baselines are flat-extrapolated outside their x range; strongly curved line ends
  can get a slightly wrong boundary.
- Crossing or severely misplaced baselines can confuse the nearest-in-y adjacency.
- grec-p5 right-folio oversegmentation (68 predicted vs 36 GT) is untouched; the
  fixer never merges or splits lines.
- The three render JPEGs are full resolution (0.6 to 0.9 MB), larger than the
  overlay.py size budget; kept simple per the brief.

## New files (all inside experiments/, plus 3 renders)

- fix_polygons.py, test_fix_polygons.py, ink_loss.py, render_polyfix.py
- out/orli-fixed-polyfix/ and out/greek-orli-folios-polyfix/ (split 0.5, which was
  the default when they were written; the default is 0.35 now)
- out/orli-fixed-polyfix-s035/, out/orli-fixed-polyfix-s065/ and the two Greek
  equivalents; out/abl-notrim, out/abl-nocap, out/abl-norepair, out/abl-noneigh
- renders: _orli-env/overlays/polygons-only/vat-1r-polyfix.jpg,
  c13-polyfix.jpg, grec-p3-polyfix.jpg

## Exact commands to reproduce

PY=/Users/krishuagarwal/Desktop/Programming/python/greekOCR/.venv/bin/python
GT=/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon
GGT=/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/greek/recon
MAN=/Users/krishuagarwal/Desktop/Programming/python/greekOCR/scripts/coptic/manifest.json
ROOT=/Users/krishuagarwal/Desktop/Programming/python/greekOCR
ENV=/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env

$PY test_fix_polygons.py  # via: $PY -m pytest test_fix_polygons.py -q
$PY fix_polygons.py --pred-dir out/orli-fixed-poly --out-dir out/orli-fixed-polyfix --split-above 0.5
$PY fix_polygons.py --pred-dir out/greek-orli-folios --out-dir out/greek-orli-folios-polyfix --split-above 0.5
$PY fix_polygons.py --pred-dir out/orli-fixed-poly --out-dir out/orli-fixed-polyfix-s035 --split-above 0.35
$PY fix_polygons.py --pred-dir out/orli-fixed-poly --out-dir out/orli-fixed-polyfix-s065 --split-above 0.65
$PY fix_polygons.py --pred-dir out/greek-orli-folios --out-dir out/greek-orli-folios-polyfix-s035 --split-above 0.35
$PY fix_polygons.py --pred-dir out/greek-orli-folios --out-dir out/greek-orli-folios-polyfix-s065 --split-above 0.65
$PY score_polygons.py --pred-dir out/orli-fixed-polyfix --gt-dir $GT --manifest $MAN --repo-root $ROOT
$PY greek_gt.py score --pred-dir out/greek-orli-folios-polyfix --gt-dir $GGT --view all --polygons
$PY ink_loss.py --orig-dir out/orli-fixed-poly --fixed-dir out/orli-fixed-polyfix --pages $ENV/pages_all10.json
$PY ink_loss.py --orig-dir out/greek-orli-folios --fixed-dir out/greek-orli-folios-polyfix --pages $ENV/pages_greek.json
$PY render_polyfix.py --slug vat-1r --image "<coptic jpg>" --before out/orli-fixed-poly/vat-1r.json --after out/orli-fixed-polyfix/vat-1r.json --out $ENV/overlays/polygons-only/vat-1r-polyfix.jpg
(same for c13 and grec-p3)
