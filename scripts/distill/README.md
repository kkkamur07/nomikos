# TrOCR logit distillation runbook

This script distils a fine-tuned TrOCR-base teacher into a 20 to 25M parameter
student with plain logit distillation (KL on soft targets plus cross-entropy
on the ground truth). It has a `smoke` mode for a quick CPU check and a `live`
mode for the full training run on the GPU box.

## Environment

Use the same environment the teacher was trained in: torch with CUDA,
transformers 5.x and pillow. The repo's uv configuration pins CPU torch
wheels on Linux, so on the GPU box use the teacher's environment
(CUDA torch, transformers 5.x, pillow) rather than `uv sync`.

## Commands

Replace `<TEACHER_DIR>` with the fine-tuned teacher checkpoint directory and
`<DATA_DIR>` with a data directory containing `image/`, `gt_train.txt`,
`gt_val.txt` and `gt_test.txt`. Repeat `--data-dir` to combine several
directories (for example `greek-ocr` and adjacent folders).

Dry run (dataset stats and parameter check, no training):

```
python scripts/distill/distill_trocr.py --teacher <TEACHER_DIR> \
  --data-dir <DATA_DIR> --out runs/distill-smoke --mode smoke --dry-run
```

Parameter check only:

```
python scripts/distill/distill_trocr.py --teacher <TEACHER_DIR> \
  --data-dir <DATA_DIR> --out runs/distill-smoke --mode smoke --param-check
```

Smoke run (about 30 steps, good for a first check on the box):

```
python scripts/distill/distill_trocr.py --teacher <TEACHER_DIR> \
  --data-dir <DATA_DIR> --out runs/distill-smoke --mode smoke
```

Live run (full 30-epoch training; run inside `tmux` or `screen`):

```
tmux new -s distill
python scripts/distill/distill_trocr.py --teacher <TEACHER_DIR> \
  --data-dir <DATA_DIR> --out runs/distill-live --mode live
```

After an interruption, resume with the same command plus `--resume`
(checkpoint state lives in `<out>/last.pt`).

Useful flags:

- `--max-val-samples N` bounds the time of each evaluation (evaluation
decodes the whole validation set greedily every `--eval-every` optimizer
steps and at each epoch end).
- `--grad-accum K` reaches a larger effective batch size (training steps
are optimizer steps).
- `--encoder-init deit-tiny` initialises the encoder from pretrained
weights (needs network once to fetch the weights).
- `--temperature` and `--alpha` tune the distillation mix.

## Reading metrics

`metrics.jsonl` holds one JSON record per line: `teacher_eval` (teacher scored
once on the val subset before training), `train` (step, lr, loss, loss_kd,
loss_ce) and `eval` (step, cer, wer, exact_match, val_loss_ce). Print the
learning curve with:

```
python -c "import json; [print(r.get('step'), round(r.get('cer', float('nan')), 4), round(r.get('wer', float('nan')), 4), round(r.get('exact_match', float('nan')), 4)) for r in map(json.loads, open('runs/distill-live/metrics.jsonl')) if r.get('type') == 'eval']"
```

## Checkpoints

`<out>/best/` holds the student with the lowest validation CER and
`<out>/final/` holds the student at the end of training. Both are standard
`save_pretrained` directories (model plus processor) and load the way the
serving stack expects:

```
VisionEncoderDecoderModel.from_pretrained("<out>/final")
TrOCRProcessor.from_pretrained("<out>/final")
```

`run_config.json` records every resolved argument and the parameter counts.

## Student shape

Defaults: encoder hidden 192, 12 layers, 3 heads; decoder dim 256, 4 layers,
4 heads, FFN 1024. Measured parameters with these defaults: total 22821312
(encoder 5738688, decoder 17082624, tied token embedding 12867840).
`--encoder-init deit-tiny` initialises the encoder from
`facebook/deit-tiny-patch16-224` (shapes must match the `--enc-*` flags) and
`--embed-init pca` initialises the token embedding from the teacher's top
principal components.

## Known limitation

No augmentation is applied during distillation; training sees the raw line
images as stored in each `--data-dir`.
