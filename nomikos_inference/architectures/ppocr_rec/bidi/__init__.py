#
# Copyright 2026 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implementation of the Unicode Bidirectional Algorithm (UAX #9) as of Unicode
17.0.0. This module is a replacement for the pre-6.3 UBA implementation from
python-bidi. This is an unoptimized, literal python port of the spec. Rules P1
and L3 are not implemented.

Vendored from kraken 7.1.1, kraken/lib/bidi, Apache License 2.0; reformatted
with ruff and split into rule modules (_common, _explicit, _weak, _neutral,
_resolve), algorithm unchanged.
"""

from nomikos_inference.architectures.ppocr_rec.bidi._common import UCD_VERSION, bidi_class
from nomikos_inference.architectures.ppocr_rec.bidi._resolve import get_display_map

__all__ = ["get_display_map", "bidi_class", "UCD_VERSION"]
