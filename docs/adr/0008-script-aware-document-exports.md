# 0008. Script-aware document exports (docx and shaped PDF)

- Status: Accepted
- Date: 2026-10-11

## Context

The platform transcribes Greek, Syriac, Armenian and Coptic manuscripts. Its transcription PDF
was drawn with reportlab's `drawString` in one bundled font, Noto Sans. That font has no Syriac,
Armenian or Coptic glyphs, and `drawString` does no shaping and no bidirectional layout, so a
Syriac line came out as nothing (or boxes, with a fallback font): Syriac is a joining script
written right to left, and drawing it one codepoint at a time left to right is wrong even with
the right font. Rendering the seeded Grec 1360 page with a Syriac, an Armenian and a Coptic line
on the old renderer drew the Greek lines and left the other three blank.

Researchers also asked for a Word file. The plain text export has the words but no font and no
direction, and Word opens a Syriac `.txt` line left to right in a Latin font.

A document carries no script or direction field, and a page can mix scripts (a Greek gloss on a
Syriac page).

## Decision

1. **Script is detected per line from Unicode blocks**
   (`backend/document/application/script_detection.py`): Syriac U+0700 to U+074F (the one
   right-to-left script), Armenian U+0530 to U+058F, Coptic U+2C80 to U+2CFF plus the Coptic
   letters U+03E2 to U+03EF, Greek U+0370 to U+03FF and U+1F00 to U+1FFF, Latin otherwise. A line
   with any Coptic-only letter counts its Greek-block letters as Coptic, because Coptic is
   commonly typed in Greek letters plus those few. Both exports use this one module.

2. **Fonts are bundled.** Noto Sans Syriac (v3.000, the Estrangela base family; the newer
   releases are the Eastern and Western variants), Noto Sans Armenian (v2.008) and Noto Sans
   Coptic (v2.004), unhinted TTFs from the notofonts releases, next to the existing Noto Sans,
   with one `OFL.txt`. `core/fonts.py::font_for_script` maps a script to its file.

3. **The PDF is drawn with fpdf2 and shaped with HarfBuzz (`uharfbuzz`)**, replacing reportlab.
   Each line is set in its script's font with Noto Sans as fallback for characters the script
   font lacks (digits, Latin), shaped with direction `rtl` for Syriac and `ltr` otherwise, and
   placed at the line's geometry as before: same page size in points, same font fitting, same
   baseline one font size below the top of the line box, same two-point padding, same colours
   for approved and unapproved text. A right-to-left line is aligned to the right edge of its
   box. Placement of the Greek lines was compared against the old renderer on the same page and
   is unchanged.

   Keeping reportlab and swapping only the text drawing was considered. reportlab has no
   supported shaping path for `drawString`; doing it by hand means running HarfBuzz ourselves and
   emitting positioned glyph ids into a reportlab font, which reimplements what fpdf2 already
   does (shaping, glyph subsetting, ToUnicode maps so the text stays extractable). fpdf2 is pure
   Python with fonttools and defusedxml as its only dependencies, and uharfbuzz ships binary
   wheels, so no system library or Dockerfile change is needed. Nothing else imported
   reportlab, so it is removed from the dependency groups.

   The per-page and the whole-document PDF now share one drawing routine (`draw_page`) instead of
   the document service restating it.

4. **A Word export** (`document_export_docx.py`, python-docx) mirrors the text export: the same
   selector, ground-truth rule and position-based page numbers, a "Page N" heading per page and a
   paragraph per line. Every run names its script font in all `w:rFonts` slots (ascii, hAnsi, cs,
   eastAsia); the document default is Noto Sans 12 pt (`sz` and `szCs`). A Syriac paragraph gets
   `w:bidi` and its run `w:rtl`. No `w:jc` is written on it: a bidi paragraph's natural alignment
   is already the right margin, and Word treats `jc` left/right on bidi paragraphs as logical
   (start/end), so writing `right` would risk the opposite. Routes:
   `GET /projects/{p}/documents/{d}/export/docx` (with `reviewed_only`) and, since the public
   surface exports page by page, `GET /public/projects/{p}/documents/{d}/parts/{part}/docx`
   behind the published-part gate and the export throttle.

## Consequences

- Syriac, Armenian and Coptic lines render with real, shaped glyphs in the PDF, and their text
  still extracts (pypdf) as the characters it draws.
- The Word file names fonts but does not embed them; a reader without the Noto families gets
  their word processor's font substitution. The families are free (OFL) and named in the file.
- The production bundle grows by fpdf2, fonttools, uharfbuzz, python-docx and lxml (several MB,
  mostly fonttools, lxml and HarfBuzz), and by about 100 KB of fonts, and loses reportlab and
  charset-normalizer.
- A new script means a block range in `script_detection.py`, a font file and two dictionary
  entries in `core/fonts.py`.
