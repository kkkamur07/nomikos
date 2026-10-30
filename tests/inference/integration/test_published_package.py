"""The published `nomikos-inference` package, exercised as an installed wheel.

Everything here runs against a real wheel installed into a real, empty virtual
environment, in a subprocess whose working directory is outside the repository.
That last detail is the whole point: run this from the repository root with the
tree importable and every assertion would pass whether or not the package
boundary exists. The subprocess proves it does.

Marked `ml` because it downloads a dependency closure and real **Hub
artifact**s, and runs both architectures on a real page - minutes, not seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.paths import REPO_ROOT, SEGMENT_PAGE, TRANSCRIBE_LINE

pytestmark = pytest.mark.ml

# ADR 0006 needs no install flag at all. Under ADR 0004 this constant was
# `--torch-backend=cpu`, and it was not a nicety: without it a Linux resolve
# pulled sixteen nvidia/triton wheels behind `torch` and this fixture alone
# installed about 4.8 GB. `onnxruntime` publishes one CPU wheel per platform,
# so there is no accelerator variant to exclude and nothing for a researcher to
# remember.

# Every platform the package claims to support. Intel macOS is back: it was
# absent under ADR 0004 only because PyTorch publishes no
# `x86_64-apple-darwin` wheel from 2.10 onward.
TARGET_PLATFORMS = (
    "x86_64-manylinux_2_28",
    "aarch64-manylinux_2_28",
    "x86_64-pc-windows-msvc",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
)


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is required to build and install the published package")
    return executable


@pytest.fixture(scope="session")
def installed_package(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the wheel and install it into an otherwise empty environment.

    Session-scoped: this costs a dependency-closure download, and every test in
    the module is asking a question about the same installed artifact.
    """
    uv = _uv()
    workspace = tmp_path_factory.mktemp("published-package")
    dist = workspace / "dist"
    venv = workspace / "venv"

    subprocess.run(
        [uv, "build", "--wheel", "-o", str(dist)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(dist.glob("nomikos_inference-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    subprocess.run(
        [uv, "venv", str(venv), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
        check=True,
        capture_output=True,
        text=True,
    )
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheels[0])],
        check=True,
        capture_output=True,
        text=True,
    )

    return {"wheel": wheels[0], "venv": venv, "python": python, "elsewhere": workspace}


def _run_installed(
    installed_package: dict[str, Path],
    source: str,
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Run `source` under the installed interpreter, outside the repository.

    `cwd` is the throwaway workspace and `PYTHONPATH` is cleared, so an import
    of `nomikos_inference` can only be satisfied by the installed wheel.

    The repository-relative settings the test session runs under are dropped
    too. `INFERENCE_REGISTRY_PATH=nomikos_inference/registry.yaml` is the load-bearing
    one: leaving it set would have the installed package read the **Registry**
    out of the checkout, and finding its own bundled copy is part of what is
    being tested.
    """
    environment = dict(os.environ)
    for leaked in ("PYTHONPATH", "INFERENCE_REGISTRY_PATH", "HF_CACHE_ROOT"):
        environment.pop(leaked, None)
    environment.update(env or {})
    completed = subprocess.run(
        [str(installed_package["python"]), "-c", source],
        cwd=installed_package["elsewhere"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _last_json_line(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])


# Five tests stood here, and between them they tested pip, hatchling and CPython's import
# system rather than this repository's code:
#
# * `..._imports_from_site_packages_not_the_repository_tree` asserted that clearing
#   `PYTHONPATH` makes an import resolve out of site-packages. That is the stdlib's
#   behaviour, and `_run_installed` arranges it for every test below anyway.
# * `test_the_console_entry_point_is_present_and_executable` asserted `[project.scripts]`
#   produced a file. `test_cli_pairing.py` and `test_cli_run.py` both assert the console
#   script exists in their `installed_cli` fixtures and then *run* it, and
#   `test_cli_pairing.py::test_the_version_subcommand_reports_the_installed_package_version`
#   keeps the `0+unknown` guard in the lane that runs on every pull request rather than
#   behind this module's `ml` marker.
# * `test_the_installed_closure_carries_no_torch_and_no_accelerator_wheels` was a strict
#   subset of `test_every_target_platform_resolves_without_an_accelerator_wheel` below,
#   which asserts the same three properties across five target platforms instead of one.
# * `test_the_installed_package_holds_no_web_server` asserted `find_spec` returns `None`
#   for `inference.api` and `inference.helper`, two modules #60 deleted; asking whether a
#   module that does not exist can be found is not a guard. Its `fastapi`/`uvicorn`/
#   `starlette` half is a closure claim, also covered five platforms wide below.
# * `test_the_installed_package_opens_no_socket` monkeypatched `socket.socket.bind` and
#   imported four modules. No library binds a socket at import time, so it could only ever
#   pass.


@pytest.fixture(scope="session")
def real_page_run(installed_package: dict[str, Path], tmp_path_factory) -> dict:
    """Segment and transcribe a real page through the installed package.

    One session-scoped run rather than one per assertion: it downloads three
    **Hub artifact**s and runs both architectures, and every question below is
    about the same execution.
    """
    cache_root = tmp_path_factory.mktemp("hub-cache")
    source = f"""
import json, pathlib
from nomikos_inference.contracts.common import InferenceTask
from nomikos_inference.jobs.runner import run_model

segment = run_model(
    task=InferenceTask.segment,
    registry_model_id="blla-segment",
    registry_tag="stable",
    image_bytes=pathlib.Path({str(SEGMENT_PAGE)!r}).read_bytes(),
)
transcribe = run_model(
    task=InferenceTask.transcribe,
    registry_model_id="syriac-calamari-v2",
    registry_tag="stable",
    image_bytes=pathlib.Path({str(TRANSCRIBE_LINE)!r}).read_bytes(),
    params={{"line_index": 0}},
)
coptic = run_model(
    task=InferenceTask.transcribe,
    registry_model_id="coptic-calamari-v1",
    registry_tag="stable",
    image_bytes=pathlib.Path({str(TRANSCRIBE_LINE)!r}).read_bytes(),
    params={{"line_index": 0}},
)
print(json.dumps({{
    "lines": len(segment.lines),
    "blocks": len(segment.blocks),
    "adapter": segment.lines[0].source_metadata.get("adapter"),
    "text": transcribe.text,
    "confidence": transcribe.confidence,
    "coptic_text": coptic.text,
    "coptic_confidence": coptic.confidence,
}}))
"""
    output = _run_installed(installed_package, source, env={"HF_CACHE_ROOT": str(cache_root)})
    return {"result": _last_json_line(output), "cache_root": cache_root}


def test_a_real_page_is_segmented_and_transcribed_through_the_installed_package(
    real_page_run: dict,
) -> None:
    result = real_page_run["result"]

    assert result["blocks"] == 1
    assert result["lines"] > 1
    assert result["adapter"] == "blla"
    assert result["text"].strip() != ""
    assert 0.0 <= result["confidence"] <= 1.0
    # The line is not Coptic script, so no text claim: the point is the third
    # artifact resolved, ran, and answered through the same runner.
    assert isinstance(result["coptic_text"], str)
    assert 0.0 <= result["coptic_confidence"] <= 1.0


# `test_the_installed_package_resolves_hf_weights_and_records_their_provenance` stood here
# and asserted the manifest's `hub_revision` and `artifact_sha256` are 40 and 64 characters
# long. That is `tests/hf/test_resolve.py`, and the test below makes the stronger claim
# about the same two manifests: not that the fields have the right *shape* but that they
# match the registry's pins and the bytes on disk.


def test_the_cached_artifacts_match_the_digests_the_registry_pins(
    real_page_run: dict,
) -> None:
    """**Artifact SHA-256**, verified independently of the code that verified it.

    Hashing the bytes here rather than trusting the manifest is what makes this
    a check on the resolver instead of a check on its own bookkeeping.
    """
    import yaml

    registry = yaml.safe_load((REPO_ROOT / "nomikos_inference" / "registry.yaml").read_text())
    cache_root = real_page_run["cache_root"]

    checked = 0
    for registry_model_id, entry in registry["models"].items():
        version = entry["versions"]["stable"]
        cache_dir = cache_root / registry_model_id / "stable"
        if not cache_dir.is_dir():
            continue
        manifest = json.loads((cache_dir / ".hub-manifest.json").read_text())
        artifact = cache_dir / manifest["artifact_path"]

        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == version["artifact_sha256"]
        assert manifest["hub_revision"] == version["hub_revision"]
        checked += 1

    assert checked == 3


# `test_a_corrupted_artifact_is_rejected_by_the_installed_verifier` stood here. Accept and
# reject for `verify_artifact_sha256` are `tests/hf/test_artifacts.py`; this ran the same
# two calls in a subprocess to show the function is in the wheel, which `real_page_run`
# above already shows by resolving three digest-pinned artifacts through it and running them.


@pytest.fixture(scope="session")
def coptic_widths_run(
    installed_package: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict:
    """The Coptic artifact through the installed session loader, at four widths.

    One session-scoped run: it resolves `coptic-calamari-v1` out of the installed
    wheel's own bundled **Registry** into its own **Hub cache**, so the download
    revision, the digest check, and the graph under test are all the shipped ones.
    """
    cache_root = tmp_path_factory.mktemp("coptic-hub-cache")
    source = """
import hashlib, json
import numpy as np
from io import BytesIO
from PIL import Image
from nomikos_inference.registry import load_registry, get_model_entry
from nomikos_inference.weights import resolve_weights_source
from nomikos_inference.architectures.calamari.adapter import _load_session
from nomikos_inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)

entry = get_model_entry(load_registry(), "coptic-calamari-v1", "stable")
version = entry.versions["stable"]
path = resolve_weights_source(
    version.weights_source,
    registry_model_id="coptic-calamari-v1",
    registry_tag="stable",
    hub_revision=version.hub_revision,
    artifact_sha256=version.artifact_sha256,
    architecture="calamari",
)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
session, charset, line_height = _load_session(str(path), None)
rng = np.random.default_rng(11)
widths = []
for width in [8, 64, 517, 1200]:
    canvas = np.full((48, width), 255, dtype=np.uint8)
    canvas[8:40, 2:max(3, width - 2)] = 0
    buffer = BytesIO()
    Image.fromarray(canvas, mode="L").save(buffer, format="PNG")
    tensor = preprocess_line_image_bytes_to_calamari_tensor(
        buffer.getvalue(), line_height=line_height
    ).astype(np.float32)
    logits, out_len = session.run(
        ["logits", "out_len"],
        {
            "image": tensor,
            "image_lengths": np.asarray([tensor.shape[1]], dtype=np.int64),
        },
    )
    widths.append({
        "width": width,
        "out_len": int(np.asarray(out_len)[0]),
        "logit_time": int(np.asarray(logits).shape[1]),
    })
print(json.dumps({
    "digest": digest,
    "line_height": line_height,
    "classes": len(charset),
    "widths": widths,
}))
"""
    output = _run_installed(installed_package, source, env={"HF_CACHE_ROOT": str(cache_root)})
    return {"result": _last_json_line(output), "cache_root": cache_root}


def test_the_coptic_artifact_runs_representative_widths(coptic_widths_run: dict) -> None:
    """The Coptic graph runs at every serving width, not just the traced one.

    The first published Coptic ONNX failed the adapter's temperature gate, and past
    that gate it ran only at the traced example width 8: every real line crashed at
    the LSTM node. So width 8 passing alongside wider widths is the point of this
    test, not an arbitrary small input: a frozen time axis passes 8 alone and fails
    everything else. `out_len` growing with width is the shape-level proof the time
    axis is dynamic end to end.
    """
    import yaml

    result = coptic_widths_run["result"]
    pin = yaml.safe_load((REPO_ROOT / "nomikos_inference" / "registry.yaml").read_text())["models"][
        "coptic-calamari-v1"
    ]["versions"]["stable"]

    assert result["digest"] == pin["artifact_sha256"]
    assert result["line_height"] == 48
    assert result["classes"] == 39
    assert [run["width"] for run in result["widths"]] == [8, 64, 517, 1200]
    out_lens = [run["out_len"] for run in result["widths"]]
    assert out_lens == sorted(out_lens) and len(set(out_lens)) == len(out_lens)
    for run in result["widths"]:
        assert run["out_len"] == run["logit_time"]


@pytest.fixture(scope="session")
def ppocr_pages_run(
    installed_package: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict:
    """The ppocr-det artifact through the installed runner, at two page sizes.

    One session-scoped run: it resolves `ppocr-segment` out of the installed
    wheel's own bundled **Registry** into its own **Hub cache**, so the download
    revision, the digest check, and the graph under test are all the shipped ones.
    The pages are synthetic white canvases with three rendered text lines; the
    second size is not a multiple of 32, so the 32-rounding resize path is
    exercised too.
    """
    cache_root = tmp_path_factory.mktemp("ppocr-hub-cache")
    source = """
import hashlib, json
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from nomikos_inference.contracts.common import InferenceTask
from nomikos_inference.jobs.runner import run_model
from nomikos_inference.registry import load_registry, get_model_entry
from nomikos_inference.weights import resolve_weights_source

entry = get_model_entry(load_registry(), "ppocr-segment", "stable")
version = entry.versions["stable"]
path = resolve_weights_source(
    version.weights_source,
    registry_model_id="ppocr-segment",
    registry_tag="stable",
    hub_revision=version.hub_revision,
    artifact_sha256=version.artifact_sha256,
    architecture="ppocr-det",
)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
# Rendered text, not solid bars: the detector answers strokes, and a solid
# black rectangle scores a max probability near 0.007, so bars alone
# detect nothing.
font = ImageFont.load_default(size=28)
pages = []
for width, height in [(480, 320), (317, 205)]:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for top in (40, 120, 200):
        draw.text((40, top), "the quick brown fox jumps over", font=font, fill="black")
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    response = run_model(
        task=InferenceTask.segment,
        registry_model_id="ppocr-segment",
        registry_tag="stable",
        image_bytes=buffer.getvalue(),
    )
    pages.append({
        "width": width,
        "height": height,
        "blocks": len(response.blocks),
        "lines": len(response.lines),
    })
print(json.dumps({
    "digest": digest,
    "hub_revision": version.hub_revision,
    "pages": pages,
}))
"""
    output = _run_installed(installed_package, source, env={"HF_CACHE_ROOT": str(cache_root)})
    return {"result": _last_json_line(output), "cache_root": cache_root}


def test_the_ppocr_det_artifact_segments_synthetic_pages(ppocr_pages_run: dict) -> None:
    """The published detector answers through the installed runner.

    The entry resolves from the wheel's bundled **Registry**, the downloaded
    bytes match the pinned digest, and both page sizes (one off the 32 grid)
    come back as a valid segment response with at least one line.
    """
    import yaml

    result = ppocr_pages_run["result"]
    pin = yaml.safe_load((REPO_ROOT / "nomikos_inference" / "registry.yaml").read_text())["models"][
        "ppocr-segment"
    ]["versions"]["stable"]

    assert result["digest"] == pin["artifact_sha256"]
    assert result["hub_revision"] == pin["hub_revision"]
    assert [page["width"] for page in result["pages"]] == [480, 317]
    for page in result["pages"]:
        assert page["blocks"] == 1
        assert page["lines"] >= 1


@pytest.mark.parametrize("platform", TARGET_PLATFORMS)
def test_every_target_platform_resolves_without_an_accelerator_wheel(
    installed_package: dict[str, Path], tmp_path: Path, platform: str
) -> None:
    """The closure has to hold where the package lands, not where it was built.

    This machine's resolution proves nothing on its own: it was Linux that
    dragged CUDA under ADR 0004, and it did so with no flag passed. Resolving
    with **no flag at all** is the claim now - the requirements come from the
    built wheel's own metadata, so the check cannot drift from what ships.
    """
    uv = _uv()
    output = _run_installed(
        installed_package,
        "import json; from importlib.metadata import metadata;"
        " print(json.dumps(metadata('nomikos-inference').get_all('Requires-Dist')))",
    )
    requirements = tmp_path / f"{platform}.in"
    requirements.write_text("\n".join(json.loads(output.strip().splitlines()[-1])) + "\n")
    resolution = tmp_path / f"{platform}.txt"

    resolved = subprocess.run(
        [
            uv,
            "pip",
            "compile",
            "--quiet",
            "--python-platform",
            platform,
            "--python-version",
            "3.11",
            str(requirements),
            "-o",
            str(resolution),
        ],
        capture_output=True,
        text=True,
    )

    assert resolved.returncode == 0, resolved.stderr
    pinned = [line.split("==")[0] for line in resolution.read_text().splitlines() if "==" in line]

    assert [name for name in pinned if name.startswith(("nvidia", "triton"))] == []
    assert [name for name in pinned if name.lower() in {"torch", "torchvision"}] == []
    assert "onnxruntime" in pinned
