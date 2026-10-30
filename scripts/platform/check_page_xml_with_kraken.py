#!/usr/bin/env python3
"""Load a folder of exported PAGE XML with kraken's own parser and report what it sees.

The unit tests answer "is this file well formed"; only kraken answers "does kraken
agree", and the two are not the same question. ``kraken.lib.xml.XMLPage`` is the
class ``ketos segtrain -f page`` builds its dataset from, so a file it reads
without complaint, with the line and region counts we expect, is a file that will
train.

kraken is deliberately not a dependency of this repository (it drags the whole
training stack, and ADR 0002 keeps the published wheel to what *runs* a model),
so the import happens inside ``main`` and this script is expected to be run from
an environment that has kraken 7 installed. Nothing else here imports it, and
nothing in ``nomikos/`` imports this.

    python scripts/platform/check_page_xml_with_kraken.py path/to/exported-xml/

Exits non-zero if any file fails to load or any line comes back without a
baseline or a boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder of exported PAGE XML files")
    parser.add_argument(
        "--glob",
        default="*.xml",
        help="which files in the folder to load (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Imported here, not at module scope: the repository does not depend on kraken.
    from kraken.lib.xml import XMLPage

    paths = sorted(args.folder.glob(args.glob))
    if not paths:
        print(f"no files matching {args.glob!r} in {args.folder}", file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        try:
            page = XMLPage(path)
        except Exception as error:  # noqa: BLE001 - any parser failure is the finding
            print(f"{path.name}: FAILED to load: {type(error).__name__}: {error}")
            failures += 1
            continue

        lines = list(page.lines.values())
        no_baseline = [
            line_id for line_id, line in page.lines.items() if not getattr(line, "baseline", None)
        ]
        no_boundary = [
            line_id for line_id, line in page.lines.items() if not getattr(line, "boundary", None)
        ]
        print(
            f"{path.name}: {len(page.regions)} region type(s), "
            f"{sum(len(v) for v in page.regions.values())} region(s), {len(lines)} line(s), "
            f"image={page.imagename.name if page.imagename else '<none>'}"
        )
        for line_id in no_baseline:
            print(f"  no baseline: {line_id}")
        for line_id in no_boundary:
            print(f"  no boundary: {line_id}")
        failures += bool(no_baseline or no_boundary)

    print(f"\n{len(paths)} file(s), {failures} with findings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
