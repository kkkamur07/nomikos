---
pretty_name: Armenian Manuscript HTR (BnF Armenien 172 and UCLA Armenian MS 72)
license: other
license_name: nomikos-mixed
license_link: LICENSE.md
task_categories:
- image-to-text
language:
- xcl
tags:
- htr
- handwritten-text-recognition
- manuscripts
- page-xml
- legal-history
- armenian
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

This release contains handwritten text recognition ground truth for two Armenian canon law manuscripts: 1,704 transcribed lines with line polygons and baselines across 34 pages. Page images are shipped for the BnF manuscript under Gallica terms. The UCLA manuscript ships as geometry and text only, pending rights confirmation.

## The language and the manuscripts

Armenian is an Indo-European language, attested in writing since the early 5th century AD, when its alphabet was invented. The alphabet held 36 letters, then 38 from the Middle Ages, and letter shapes changed over time through what can be described as majuscule, minuscule and cursive forms. For the period NOMOS covers, Armenian juridical manuscripts concern almost entirely canon law, the law of the Armenian Church. The "Armenian Book of Canons", the main text of Armenian canon law, was created around the 7th century. Its manuscripts date from the 11th century onward, most of them from the 15th century or later, and more than 200 survive today.

NOMIKOS was trained on three manuscripts in two scripts, minuscule and cursive. No manuscripts with legal content are known in majuscule proper. The manuscripts were traced through the online catalogue of Armenian digitized manuscripts (https://www.armenian-manuscripts-index.com/) and chosen for image quality and script variety:

1. Paris, Bibliothèque nationale de France, ms. Arm. 331, folia 24r to 34r, text in one column. This manuscript is background to the training work and is not part of this release.
2. Paris, Bibliothèque nationale de France, ms. Arm. 172, folia 215v to 223v, text in two columns. Included here.
3. Los Angeles, UCLA Library Special Collections, Armenian MS 72, folia 64v to 73v, text in two columns. Included here.

As the project continues, transcriptions will grow in length and variety, with more image sources and more complex layouts such as marginal notes and interlinear additions.

## What is in this release

| Manuscript | Pages | Paired lines | Images shipped |
| --- | ---: | ---: | --- |
| Paris, BnF, Arménien 172, ff. 215v to 223v | 17 | 886 | yes |
| Los Angeles, UCLA, Armenian MS 72, ff. 64v to 73v | 17 | 818 | no |

Only pages with at least one paired line are included. Eighteen segments across the two manuscripts have no paired text yet and appear only in the page level objects. Every number in this card comes from manifest.csv.

## Files and formats

Each manuscript folder holds native PAGE XML 2019 files (`page-xml/<manuscript>/NNNN.xml`), with page images beside them (`images/<manuscript>/NNNN.<ext>`) where images are shipped, so the pages open directly in eScriptorium, Kraken and Transkribus. Two Parquet configs carry the same content in machine readable form. The `lines` config has one row per paired line: `line_id`, `manuscript`, `shelfmark`, `page_index`, `line_index` in reading order, `text`, `image` (the line crop, cut to the axis-aligned bounding box of the polygon with 4 px padding and stored as PNG, null where images are not shipped), `polygon`, `baseline`, `bbox`, `page_width`, `page_height`, `image_shipped` and `source_url`. The `pages` config has one row per page with the full page image (null where images are not shipped) and an `objects` column holding every segment, paired or not. There is a single `train` split. The corpus is too small for a validation or test split to be meaningful.

```python
from datasets import load_dataset
lines = load_dataset("nomikos-project/armenian-manuscript-htr", "lines", split="train")
pages = load_dataset("nomikos-project/armenian-manuscript-htr", "pages", split="train")
```

## Images and rights

The Arménien 172 images are digital reproductions from Gallica, the digital library of the Bibliothèque nationale de France. Gallica's conditions of use (https://gallica.bnf.fr/accueil/fr/html/conditions-dutilisation-de-gallica) state that most Gallica content reproduces public domain holdings, and that "La réutilisation non commerciale de ces contenus est libre et gratuite dans le respect de la législation en vigueur et notamment du maintien de la mention de source" ("Source gallica.bnf.fr / Bibliothèque nationale de France"). Commercial reuse, including resale of the content or reuse generating direct revenue, is subject to payment and needs a licence (https://www.bnf.fr/fr/faire-une-utilisation-commerciale-dune-reproduction). Researchers are exempt from those fees for scientific publications. The required credit line is:

Source gallica.bnf.fr / Bibliothèque nationale de France

These are Gallica's own terms, not a Creative Commons licence. Before publication, the per-document notice on the manuscript's own Gallica viewer page should be re-checked. The direct manuscript viewer link (ARK) is not confirmed yet and is listed under the open questions in SUMMARY.md of the build.

No images ship for UCLA Armenian MS 72. UCLA Library places clearance responsibility on the researcher: "Document reproduction, streaming, or dissemination beyond the scope allowed under traditional academic Fair Use principles requires explicit, written authorization", and "The final legal responsibility for identifying rights holders, securing permissions, and addressing property restrictions rests exclusively with the researcher" (https://guides.library.ucla.edu/c.php?g=1423298&p=10570712). Its metadata statement releases descriptions under CC0 but explicitly excludes images and other digital content. Image release waits on a direct inquiry to copyright@library.ucla.edu confirming the item's rights status. Until then, fetch the images from the UCLA catalogue record at https://digital.library.ucla.edu/catalog/ark:/21198/zz0026w379. Coordinates in this release refer to catalogue images of 3576 to 3640 by 4688 to 4728 pixels; per-page sizes are in manifest.csv.

## Licence

Transcriptions, line polygons and baselines are the project's own work and are released under CC BY 4.0, with credit to "NOMOS project, LMU Munich". Page images stay under the holding library's terms described above and are never covered by the CC licence. See LICENSE.md for the full two-part text.

## How the data was made

The pages were segmented and transcribed on the NOMIKOS platform (https://nomikos.app), the HTR platform of the NOMOS project. Automatic line detection was corrected by hand, and the project's scholars typed the transcriptions. A line is included only when its text is paired to a line polygon. Lines follow transcription reading order in the PAGE XML.

## Known limits

The release holds 1,704 lines on 34 pages, in one hand per manuscript. There is only a train split. Eighteen segments have no paired text yet and appear only in the page level objects. BnF Arménien 331, used in training, is not included.

## The NOMOS project

NOMIKOS is the HTR platform of the ERC Consolidator Grant NOMOS, "The Challenge of Islam and the Transformation of Eastern Christian Normative Regimes, ca. 630-1100", grant 101171233, at Ludwig-Maximilians-Universität München, with PI Prof. Dr. Zachary Chitwood. See https://cordis.europa.eu/project/id/101171233 and http://nomosproject.de/ and the platform at https://nomikos.app.

## Citation

```bibtex
@misc{nomos_armenian_manuscript_htr_2026,
  author = {NOMOS project},
  title = {Armenian Manuscript HTR: canon law manuscripts line transcriptions and page layout},
  year = {2026},
  url = {https://huggingface.co/datasets/nomikos-project/armenian-manuscript-htr}
}
```

## Contact

Questions about the data can go through the NOMOS project site at http://nomosproject.de/ or the NOMIKOS platform at https://nomikos.app.
