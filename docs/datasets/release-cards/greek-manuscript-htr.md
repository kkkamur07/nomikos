---
pretty_name: Byzantine Greek Manuscript HTR (BnF Grec 1360)
license: other
license_name: nomikos-mixed
license_link: LICENSE.md
task_categories:
- image-to-text
language:
- grc
tags:
- htr
- handwritten-text-recognition
- manuscripts
- page-xml
- legal-history
- greek
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

# Byzantine Greek Manuscript HTR (BnF Grec 1360)

This release contains handwritten text recognition ground truth for five pages of Paris, Bibliothèque nationale de France, Grec 1360, a 1351 copy of the Hexabiblos of Constantine Harmenopoulos: 204 transcribed lines with line polygons and baselines, plus the five page images.

## The language and the manuscripts

Byzantine or medieval Greek was the principal language of the East Roman state and the Orthodox Church through the Middle Ages. Much of the philosophical and scientific heritage of antiquity survives in manuscripts copied by Byzantine scribes. Roughly 55,000 Greek manuscripts exist worldwide.

The NOMOS project works with manuscripts of the Hexabiblos ("Six Books") of the fourteenth-century Byzantine judge and legal scholar Constantine Harmenopoulos, given the project's legal-historical focus. The Hexabiblos gathers a broad selection of Byzantine secular law and survives in many copies. Pinakes, the standard database of Greek manuscripts, records it in whole or in part in 112 manuscripts, of which 68 hold the complete text. The main manuscript transcribed so far is Bibliothèque nationale de France, Grec 1360, copied in 1351. As the project continues, transcription will extend to other Hexabiblos manuscripts, including their scholia. Those further manuscripts are not part of this release.

## What is in this release

| Manuscript | Pages | Paired lines | Images shipped |
| --- | ---: | ---: | --- |
| Paris, BnF, Grec 1360 | 5 | 204 | yes |

Only pages with at least one paired line are included. The text of spreads 1 to 3 comes from an earlier NOMIKOS PAGE XML export of the same transcription and spreads 4 and 5 from the live platform, and `text_source` records this per page. One included page keeps 30 unpaired segments alongside its 32 paired lines. Every number in this card comes from manifest.csv.

| Page | Folio | Paired lines | Text source |
| --- | --- | ---: | --- |
| 0000 | 1r | 23 | local-page-xml |
| 0001 | 1v-2r | 51 | local-page-xml |
| 0002 | 2v-3r | 49 | local-page-xml |
| 0003 | 3v-4r | 49 | production |
| 0004 | 4v-5r | 32 | production |

## Files and formats

Each manuscript folder holds native PAGE XML 2019 files (`page-xml/<manuscript>/NNNN.xml`) next to the page images (`images/<manuscript>/NNNN.<ext>`), so the pages open directly in eScriptorium, Kraken and Transkribus. Two Parquet configs carry the same content in machine readable form. The `lines` config has one row per paired line: `line_id`, `manuscript`, `shelfmark`, `page_index`, `line_index` in reading order, `text`, `image` (the line crop, cut to the axis-aligned bounding box of the polygon with 4 px padding and stored as PNG), `polygon`, `baseline`, `bbox`, `page_width`, `page_height`, `image_shipped`, `source_url`, `folio` and `text_source`. The `pages` config has one row per page with the full page image and an `objects` column holding every segment, paired or not. There is a single `train` split. The corpus is too small for a validation or test split to be meaningful.

```python
from datasets import load_dataset
lines = load_dataset("nomikos-project/greek-manuscript-htr", "lines", split="train")
pages = load_dataset("nomikos-project/greek-manuscript-htr", "pages", split="train")
```

## Images and rights

The page images are digital reproductions from Gallica, the digital library of the Bibliothèque nationale de France. Gallica's conditions of use (https://gallica.bnf.fr/accueil/fr/html/conditions-dutilisation-de-gallica) state that most Gallica content reproduces public domain holdings, and that "La réutilisation non commerciale de ces contenus est libre et gratuite dans le respect de la législation en vigueur et notamment du maintien de la mention de source" ("Source gallica.bnf.fr / Bibliothèque nationale de France"). Commercial reuse, including resale of the content or reuse generating direct revenue, is subject to payment and needs a licence (https://www.bnf.fr/fr/faire-une-utilisation-commerciale-dune-reproduction). Researchers are exempt from those fees for scientific publications. Gallica's terms also cover machine learning. The use of Gallica documents and data "dans le cadre d'un projet en Intelligence artificielle (IA) ayant pour objet des traitements de données par l'IA (notamment la fouille de données) ou de l'IA générative" ("as part of an Artificial Intelligence (AI) project whose object is AI data processing (notably data mining) or generative AI") is "gratuite à des fins de recherche scientifique académique, à but non lucratif, sous réserve de la mention de source : BnF / Gallica" ("free for academic scientific research purposes, non-profit, subject to the source mention: BnF / Gallica") and "Payante pour les autres utilisations. Celles-ci font alors l'objet d'une autorisation et d'une licence." ("Paid for other uses. These then require an authorisation and a licence.", contact given by the BnF: prestations.numeriques@bnf.fr). Anyone may train on the transcriptions and geometry under CC BY 4.0, but training on the page images or line crops outside non-profit academic research needs a licence from the BnF. The required credit line is:

Source gallica.bnf.fr / Bibliothèque nationale de France

These are Gallica's own terms, not a Creative Commons licence. The manuscript can be found on Gallica (https://gallica.bnf.fr/) by its shelfmark.

## Licence

Transcriptions, line polygons and baselines are the project's own work and are released under CC BY 4.0, with credit to "Prof. Dr. Zachary Chitwood, NOMOS project, LMU Munich". The page images stay under the holding library's terms described above and are never covered by the CC licence. For machine learning uses of the page images, Gallica grants free use for non-profit academic scientific research with the credit "BnF / Gallica" and requires a paid licence otherwise, while general reuse carries the credit "Source gallica.bnf.fr / Bibliothèque nationale de France". See LICENSE.md for the full two-part text.

## How the data was made

The pages were segmented and transcribed on the NOMIKOS platform (https://nomikos.app), the HTR platform of the NOMOS project. Automatic line detection was corrected by hand, and the project's scholars typed the transcriptions. A line is included only when its text is paired to a line polygon. Lines follow transcription reading order in the PAGE XML.

## Known limits

The release is small: 204 lines on 5 pages, in one hand. There is only a train split. Thirty segments on one page have no paired text yet and appear only in the page level objects.

## The NOMOS project

NOMIKOS is the HTR platform of the ERC Consolidator Grant NOMOS, "The Challenge of Islam and the Transformation of Eastern Christian Normative Regimes, ca. 630-1100", grant 101171233, at Ludwig-Maximilians-Universität München, with PI Prof. Dr. Zachary Chitwood. See https://cordis.europa.eu/project/id/101171233 and http://nomosproject.de/ and the platform at https://nomikos.app.

## Credits

The ground truth transcriptions were provided by Prof. Dr. Zachary Chitwood of the NOMOS project. The NOMIKOS team segmented the pages, paired text to lines and curated the release.

## Citation

```bibtex
@misc{nomos_greek_manuscript_htr_2026,
  author = {Chitwood, Zachary and {NOMOS project}},
  title = {Byzantine Greek Manuscript HTR: BnF Grec 1360 line transcriptions and page layout},
  year = {2026},
  url = {https://huggingface.co/datasets/nomikos-project/greek-manuscript-htr}
}
```

## Contact

Questions about the data can go through the NOMOS project site at http://nomosproject.de/ or the NOMIKOS platform at https://nomikos.app.
