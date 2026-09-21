---
pretty_name: East Syriac Manuscript HTR (Alqosh 65)
license: other
license_name: nomikos-mixed
license_link: LICENSE.md
task_categories:
- image-to-text
language:
- syc
tags:
- htr
- handwritten-text-recognition
- manuscripts
- page-xml
- legal-history
- syriac
size_categories:
- n<1K
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

# East Syriac Manuscript HTR (Alqosh 65)

This release contains handwritten text recognition ground truth for seven pages of Alqosh 65, a nineteenth-century East Syriac manuscript that contains a letter by a jurist-bishop-philosopher on the ten categories of Aristotle: 143 transcribed lines with line polygons and baselines, plus the seven page images. Geometry and text are the project's own work.

## The language and the manuscripts

Syriac is the language of various Christian communities across the Eastern Mediterranean and Western Asia. Classical Syriac survives today mainly in liturgical use, and it has left a literary heritage spanning centuries. Much of it sits in manuscripts that remain unedited and unpublished to modern standards.

Classical Syriac has two principal dialects, Eastern and Western, with considerable phonological differences and, critically for recognition work, distinct scripts: Serto for Western, Madnḥaya for Eastern, alongside the older and widely used Estrangela.

The training work began with pages from an Estrangela Bible studied at a winter school in Vienna, then turned to the legal tradition that is the NOMOS focus. That background material, including pages from Vatican Syriac Borgia 82 (a nineteenth-century copy of Alqosh 169, the most significant manuscript in the East Syriac legal tradition), is not part of this release. What is released is Alqosh 65, likewise nineteenth century and in Eastern Syriac, holding a letter by a jurist-bishop-philosopher on the ten categories of Aristotle.

## What is in this release

| Manuscript | Pages | Paired lines | Images shipped |
| --- | ---: | ---: | --- |
| Alqosh, Chaldean collection, MS 65 | 7 | 143 | yes |

Only pages with at least one paired line are included. Every segment on these seven pages is paired, so the line and page level counts agree exactly. Every number in this card comes from manifest.csv.

## Files and formats

The manuscript folder holds native PAGE XML 2019 files (`page-xml/<manuscript>/NNNN.xml`) next to the page images (`images/<manuscript>/NNNN.<ext>`), so the pages open directly in eScriptorium, Kraken and Transkribus. Two Parquet configs carry the same content in machine readable form. The `lines` config has one row per paired line: `line_id`, `manuscript`, `shelfmark`, `page_index`, `line_index` in reading order, `text`, `image` (the line crop, cut to the axis-aligned bounding box of the polygon with 4 px padding and stored as PNG), `polygon`, `baseline`, `bbox`, `page_width`, `page_height`, `image_shipped`, `source_url`, `folio` and `text_source`. The `pages` config has one row per page with the full page image and an `objects` column holding every segment. There is a single `train` split. The corpus is too small for a validation or test split to be meaningful.

```python
from datasets import load_dataset
lines = load_dataset("nomikos-project/syriac-manuscript-htr", "lines", split="train")
pages = load_dataset("nomikos-project/syriac-manuscript-htr", "pages", split="train")
```

## Images and rights

The seven page images of Alqosh 65 ship with this release by decision of the NOMOS project. They reproduce a nineteenth-century manuscript and remain subject to the rights of the holding collection; the CC BY 4.0 licence covers the transcriptions, polygons and baselines only, never the images. Rights holders with a concern can reach the project through the contact section. Coordinates in this release refer to page images of 2550 by 3300 pixels.

## Licence

Transcriptions, line polygons and baselines are the project's own work and are released under CC BY 4.0, with credit to "Dr. Nima Jamali, NOMOS project, LMU Munich". See LICENSE.md for the full two-part text.

## How the data was made

The pages were segmented and transcribed on the NOMIKOS platform (https://nomikos.app), the HTR platform of the NOMOS project. Automatic line detection was corrected by hand, and the project's scholars typed the transcriptions. A line is included only when its text is paired to a line polygon. Lines follow transcription reading order in the PAGE XML.

## Known limits

The release holds 143 lines on 7 pages, in one hand. There is only a train split.

## The NOMOS project

NOMIKOS is the HTR platform of the ERC Consolidator Grant NOMOS, "The Challenge of Islam and the Transformation of Eastern Christian Normative Regimes, ca. 630-1100", grant 101171233, at Ludwig-Maximilians-Universität München, with PI Prof. Dr. Zachary Chitwood. See https://cordis.europa.eu/project/id/101171233 and http://nomosproject.de/ and the platform at https://nomikos.app.

## Credits

The ground truth transcriptions were provided by Dr. Nima Jamali of the NOMOS project. The NOMIKOS team segmented the pages, paired text to lines and curated the release.

## Citation

```bibtex
@misc{nomos_syriac_manuscript_htr_2026,
  author = {Jamali, Nima and {NOMOS project}},
  title = {East Syriac Manuscript HTR: Alqosh 65 line transcriptions and page layout},
  year = {2026},
  url = {https://huggingface.co/datasets/nomikos-project/syriac-manuscript-htr}
}
```

## Contact

Questions about the data can go through the NOMOS project site at http://nomosproject.de/ or the NOMIKOS platform at https://nomikos.app.
