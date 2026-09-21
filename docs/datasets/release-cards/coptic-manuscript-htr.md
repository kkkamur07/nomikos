---
pretty_name: Coptic Manuscript HTR (Borg. copt. 109 and Naples IB.10)
license: cc-by-4.0
task_categories:
- image-to-text
language:
- cop
tags:
- htr
- handwritten-text-recognition
- manuscripts
- page-xml
- legal-history
- coptic
size_categories:
- 1K<n<10K
configs:
- config_name: lines
  default: true
  data_files:
  - split: train
    path: data/lines/train-00000-of-00001.parquet
- config_name: pages
  data_files:
  - split: train
    path: data/pages/train-00000-of-00001.parquet
---

This release contains handwritten text recognition ground truth for two witnesses of Coptic Literary Manuscript 359, a translation of the Corpus Canonum: 2,821 transcribed lines with line polygons and baselines across 33 pages. No page images ship; the Vatican forbids redistribution and the Naples terms are unconfirmed. Geometry and text are the project's own work.

## The language and the manuscripts

Coptic is the last stage of the ancient Egyptian language, written in an alphabet based on Greek letters with a few extra signs borrowed from Demotic. It flourished as the language of Egyptian Christianity between roughly the 3rd and 12th centuries CE. What survives today is largely religious, legal and literary texts copied by scribes in monasteries.

The training data comes from Coptic Literary Manuscript 359, which preserves a translation of the Corpus Canonum, the earliest canonical collection of Late Antiquity. Like many manuscripts of this kind, its leaves have scattered across libraries over the centuries. The folios transcribed here sit in two of them: the National Library of Naples (IB.10.10-24) and the Vatican Library (Borg. copt. 109, cass. XXIX, fasc. 159, 1-3).

## What is in this release

| Manuscript | Pages | Paired lines | Images shipped |
| --- | ---: | ---: | --- |
| Vatican City, BAV, Borg. copt. 109 | 18 | 1,028 | no |
| Naples, BNVE, IB.10.10-24 | 15 | 1,793 | no |

Only pages with at least one paired line are included. Four further Naples pages carry line segmentation but no transcription yet, so they are left out. Ten segments across the release have no paired text and appear only in the page level objects. Every number in this card comes from manifest.csv.

## Files and formats

Each manuscript folder holds native PAGE XML 2019 files (`page-xml/<manuscript>/NNNN.xml`), which open directly in eScriptorium, Kraken and Transkribus once the page images are fetched from the libraries (see below). Two Parquet configs carry the same content in machine readable form. The `lines` config has one row per paired line: `line_id`, `manuscript`, `shelfmark`, `page_index`, `line_index` in reading order, `text`, `image` (null throughout this release), `polygon`, `baseline`, `bbox`, `page_width`, `page_height`, `image_shipped` and `source_url`. The `pages` config has one row per page with a null image and an `objects` column holding every segment, paired or not. There is a single `train` split. The corpus is too small for a validation or test split to be meaningful.

```python
from datasets import load_dataset
lines = load_dataset("nomikos-project/coptic-manuscript-htr", "lines", split="train")
pages = load_dataset("nomikos-project/coptic-manuscript-htr", "pages", split="train")
```

## Images and rights

No page images ship with this release.

The Vatican pages sit in DigiVatLib. Its terms allow free use of images for personal use or study purposes only: "Free use of this image is only for personal use or study purposes." Rights must be requested for "any use in printed or online publications", and "Neither the text nor the images may be reproduced, in any form, without the authorisation of the Vatican Library, 00120, Vatican City." See https://www.vaticanlibrary.va/en/information-for-readers/photographic-reproductions.html and the viewer at https://digi.vatlib.it/view/MSS_Borg.copt.109. Image release would need a written request to rights@vatlib.it. Coordinates in this release refer to page images of 1136 by 1348 pixels.

For Naples, no confirmed image host or item level terms were found, so no redistribution is assumed. The Italian cultural heritage code governs reproduction permission and fees rather than granting open redistribution, and its scope for bulk open publication is unsettled. Image release waits on direct confirmation from the Biblioteca Nazionale "Vittorio Emanuele III" of where the IB.10.10-24 images live and on what terms. Coordinates in this release refer to page images between 1310 and 1462 pixels wide and 847 and 982 pixels high; per-page sizes are in manifest.csv. The catalogue reference is CLM 359, IB.10.10-24.

## Licence

Transcriptions, line polygons and baselines are the project's own work and are released under CC BY 4.0, with credit to "NOMOS project, LMU Munich".

## How the data was made

The pages were segmented and transcribed on the NOMIKOS platform (https://nomikos.app), the HTR platform of the NOMOS project. Automatic line detection was corrected by hand, and the project's scholars typed the transcriptions. A line is included only when its text is paired to a line polygon. Lines follow transcription reading order in the PAGE XML.

## Known limits

The release holds 2,821 lines on 33 pages, in one hand per manuscript. There is only a train split. Ten segments have no paired text yet and appear only in the page level objects. Four further Naples pages are excluded until their transcription is done. There are no page images in the download; any model training needs the images fetched separately from the holding libraries first.

## The NOMOS project

NOMIKOS is the HTR platform of the ERC Consolidator Grant NOMOS, "The Challenge of Islam and the Transformation of Eastern Christian Normative Regimes, ca. 630-1100", grant 101171233, at Ludwig-Maximilians-Universität München, with PI Prof. Dr. Zachary Chitwood. See https://cordis.europa.eu/project/id/101171233 and http://nomosproject.de/ and the platform at https://nomikos.app.

## Citation

```bibtex
@misc{nomos_coptic_manuscript_htr_2026,
  author = {NOMOS project},
  title = {Coptic Manuscript HTR: CLM 359 witnesses line transcriptions and page layout},
  year = {2026},
  url = {https://huggingface.co/datasets/nomikos-project/coptic-manuscript-htr}
}
```

## Contact

Questions about the data can go through the NOMOS project site at http://nomosproject.de/ or the NOMIKOS platform at https://nomikos.app.
