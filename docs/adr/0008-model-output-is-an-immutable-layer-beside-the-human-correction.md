# 0008. Model output is an immutable layer beside the human correction

- Status: Accepted
- Date: 2026-09-18

## Context

The platform's purpose is a correction loop: a model proposes, a person fixes, and the fixes
become training data. That only works if, for any page, the database can still say what the model
proposed and what the person ended with. Transcription already worked this way. Each recogniser
run writes its own `Transcription` layer of kind `model`, and human text lives in the single
`ground_truth` layer, so the two are side by side and the correction is a diff.

Segmentation did not. Every geometry edit overwrote the line in place and flipped `source` to
manual, deletes were hard deletes, and re-segmentation dropped untouched lines. Once a person had
worked a page, the model's proposal for it no longer existed anywhere. The most valuable signal in
the loop, the false positives a person deleted, left no trace at all: a deleted line is simply a
row that is gone.

Neither side recorded which model did the proposing in a way that survives. `Job.model_id` and
`Transcription.created_by_job_id` are both `ON DELETE SET NULL`, and users may delete finished
jobs and clear a project's job history. Clearing history therefore erased the identity of the
model behind every layer on the project. The inference callback itself
(`JobCallbackRequest`) carries no model identity at all: only `inference_job_id`,
`product_job_id`, `task`, `status`, `output`, `error`. Identity has to be read from the `Job` row
at merge time or it is not recoverable later.

## Decision

**A segmentation run is an immutable row, written in the same transaction as the lines it
produced.** `segmentation_runs` stores the canonical output verbatim as JSONB (`lines`,
`blocks`), the page size it was computed against, and the model identity. It is written by
`SegmentMergeService.apply_sync` before anything is deleted, so the proposal is durable whatever
happens to the live rows afterwards. Nothing ever updates a run.

**Retention: keep every run.** A run is a few kilobytes of JSONB per page and is the only record
of what was proposed; a run whose lines have all been corrected away is exactly the run worth
keeping. Re-segmentation adds a run, it does not replace one. Pruning, if it is ever needed, is a
separate decision with a separate record.

**Live lines carry their origin in real columns**, `lines.segmentation_run_id` and
`lines.run_external_id`, not as keys in `source_metadata`. Three reasons: the pair is the diff's
join key and deserves an index; `source_metadata` is client-writable through
`LineUpsertRequest`, so a bulk save could erase the provenance it is meant to prove; and in-place
mutation (patch, merge, trim) preserves real columns for free, where a JSONB blob has to be
rewritten carefully by every path. Splits copy the origin onto both pieces, so a split reads as
one correction with two live ids.

**Model identity is denormalised at write time**, onto `segmentation_runs` and onto
`Transcription`: name, provider, artifact ref, and the registry id and tag split out of it, plus
the effective `ml_params` merged at submission. The FK to the job stays, and stays nullable. The
denormalised copy is what survives job deletion.

**The ground-truth line records what it was accepted from.**
`line_transcriptions.seeded_from_transcription_id` is set by `copy_to_ground_truth` and is not
touched by later text edits, which write only `text` and `confidence`. So a ground-truth line
says which model layer a person started from, however many times they have since retyped it.

**The correction is derived, never stored.** `diff_segmentation_run` is a pure function over the
run's JSONB and the live rows, returning unchanged / corrected / deleted / added. There is no
`corrections` table and no trigger: a stored diff is a second source of truth that goes stale the
moment someone edits a polygon.

## Deliberately not stored

- **Per-edit history.** `lines.updated_at` and `updated_by` say that a line was changed and by
  whom, not what it looked like in between. The pair that matters for training is the model's
  proposal and the current state; the intermediate drags are not training data.
- **A snapshot of model text per line.** Model `LineTranscription` rows are deleted with their
  `Line` on re-segmentation or delete, so the text a model proposed for a line that no longer
  exists is lost. This is a known gap, recorded here rather than fixed: the transcription layer
  is per line, and making it survive its line means giving it the run's shape. If measuring
  recogniser output on pages that were later re-segmented turns out to matter, that is the fix,
  and it is a separate record.
- **Pre-existing provenance.** The new columns are nullable with no backfill. A page segmented
  before this migration has no run, and the diff endpoint says so rather than inventing one.
  `scripts/platform/backfill_segmentation_runs.py` can reconstruct identity from surviving job
  rows, but not geometry, which was never written down.
- **`reviewed_at` / `reviewed_by` history.** Both are cleared when a page is unmarked, so
  `reviewed = false` can never be read beside a name that seems to contradict it.

## Consequences

Migration `007_model_output_provenance` is additive: one new table, and nullable columns with no
defaults that would rewrite `lines`, `transcriptions`, `line_transcriptions` or `document_parts`.
It is safe on a populated database and the downgrade is destructive by nature, since the runs it
drops cannot be reconstructed.

Every path that mutates geometry now has to preserve the origin columns, and each has a test that
fails if it stops: patch, bulk replace, split, merge, trim, delete, segment-health delete, and
re-segmentation. A new geometry path that forgets them does not break anything loudly; it just
makes its lines read as `added`. That is the maintenance cost this decision accepts.
