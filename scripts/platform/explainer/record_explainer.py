#!/usr/bin/env python3
"""Record the silent, subtitled Nomikos videos with Playwright.

Two videos, picked with ``--video``:

* ``tour``: a tour of under a minute of what changed in the latest round,
  logged in as the dev user on the seeded Grec 1360 document.
* ``explainer``: the walk through a new user meets, under two minutes:
  register, project, document, upload, a segmentation job, the editor on a
  transcribed document, an export, sharing and the public read only page.

Captions come from ``captions.json`` next to this file and are burned into
the picture: a fixed caption bar is injected into the page, so the recording
needs no post-production. A small circle follows the mouse because Playwright
does not draw the cursor. The webm is converted to mp4 with ffmpeg and a
``captions*.md`` file with timestamps is written beside it.

Everything that depends on the stack comes from environment variables, see
README.md. The script is re-runnable: it removes what an earlier run left
behind (Eleni's project, the 4v region, open jobs) through the API before
recording.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Locator, Page, sync_playwright

HERE = Path(__file__).resolve().parent

BASE_URL = os.environ.get("NOMIKOS_BASE_URL", "http://localhost:3160").rstrip("/")
API_URL = os.environ.get("NOMIKOS_API_URL", "http://localhost:8160").rstrip("/")
DEV_EMAIL = os.environ.get("NOMIKOS_DEV_EMAIL", "dev@example.com")
DEV_PASSWORD = os.environ.get("NOMIKOS_DEV_PASSWORD", "dev-pass-123")
USER_EMAIL = os.environ.get("NOMIKOS_USER_EMAIL", "eleni.markou@example.org")
USER_NAME = os.environ.get("NOMIKOS_USER_NAME", "eleni.markou")
USER_PASSWORD = os.environ.get("NOMIKOS_USER_PASSWORD", "Athos-Letters-2026")
SECOND_EMAIL = os.environ.get("NOMIKOS_SECOND_EMAIL", "nikos.papadakis@example.org")
SECOND_NAME = os.environ.get("NOMIKOS_SECOND_NAME", "nikos.papadakis")
SECOND_PASSWORD = os.environ.get("NOMIKOS_SECOND_PASSWORD", "nikos-pass-123")
SEED_PROJECT = os.environ.get("NOMIKOS_SEED_PROJECT", "Byzantine Greek Manuscripts")
SEED_DOCUMENT = os.environ.get("NOMIKOS_SEED_DOCUMENT", "Grec 1360")
PROJECT_NAME = "Athos Letters"
DOCUMENT_NAME = "Codex Athous 17"
IMAGE_DIR = Path(
    os.environ.get(
        "NOMIKOS_IMAGE_DIR",
        "/Users/krishuagarwal/Desktop/Programming/python/greekOCR/data/annotated/data/manuscripts/pages",
    )
)
IMAGES = os.environ.get(
    "NOMIKOS_IMAGES",
    "Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_7.jpeg,"
    "Grec_1360_CONSTANTINUS_Harmenopulus_btv1b10721710m_8.jpeg",
).split(",")
OUT_DIR = Path(os.environ.get("NOMIKOS_VIDEO_OUT", str(HERE / "out")))
FFMPEG = os.environ.get("FFMPEG", "/opt/homebrew/bin/ffmpeg")
SLOW_MO = int(os.environ.get("NOMIKOS_SLOW_MO", "150"))
VIEWPORT = {"width": 1440, "height": 900}
MIN_HOLD = 2.5

# Injected into every page: the caption bar (text kept in sessionStorage so it
# survives navigations), the cursor dot, and a rule hiding the Next.js dev
# indicator, which is not part of the product.
INIT_SCRIPT = """
(() => {
  const install = () => {
    if (!document.body || document.getElementById('__cap')) return;
    const style = document.createElement('style');
    style.textContent = `
      nextjs-portal { display: none !important; }
      #__cap { position: fixed; left: 50%; bottom: 96px; transform: translateX(-50%);
        z-index: 2147483647; max-width: 980px; padding: 12px 26px; border-radius: 10px;
        background: rgba(15, 18, 24, 0.82); color: #fff; font: 500 22px/1.35 -apple-system,
        "Segoe UI", Helvetica, Arial, sans-serif; text-align: center; pointer-events: none;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
      #__cap:empty { display: none; }
      #__cur { position: fixed; z-index: 2147483646; width: 22px; height: 22px; margin: -11px 0 0 -11px;
        border-radius: 50%; background: rgba(239, 68, 68, 0.35); border: 2px solid rgba(220, 38, 38, 0.9);
        pointer-events: none; left: -50px; top: -50px; transition: transform 0.12s; }
      #__cur.down { transform: scale(0.7); background: rgba(220, 38, 38, 0.6); }`;
    document.head.appendChild(style);
    const cap = document.createElement('div');
    cap.id = '__cap';
    try { cap.textContent = sessionStorage.getItem('__cap') || ''; } catch (e) {}
    document.body.appendChild(cap);
    const cur = document.createElement('div');
    cur.id = '__cur';
    try {
      const pos = JSON.parse(sessionStorage.getItem('__cur') || 'null');
      if (pos) { cur.style.left = pos[0] + 'px'; cur.style.top = pos[1] + 'px'; }
    } catch (e) {}
    document.body.appendChild(cur);
    const move = (e) => {
      cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px';
      try { sessionStorage.setItem('__cur', JSON.stringify([e.clientX, e.clientY])); } catch (err) {}
    };
    window.addEventListener('mousemove', move, true);
    window.addEventListener('pointermove', move, true);
    window.addEventListener('mousedown', () => cur.classList.add('down'), true);
    window.addEventListener('mouseup', () => cur.classList.remove('down'), true);
  };
  window.__setCaption = (text) => {
    try { sessionStorage.setItem('__cap', text); } catch (e) {}
    install();
    const cap = document.getElementById('__cap');
    if (cap) cap.textContent = text;
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install);
  else install();
  new MutationObserver(install).observe(document.documentElement, { childList: true, subtree: true });
})();
"""


# --- API helpers (stdlib only) -------------------------------------------


class StepFailed(RuntimeError):
    """A precondition of the recording does not hold."""


def need(condition: object, message: str) -> None:
    if not condition:
        raise StepFailed(message)


need(API_URL.startswith(("http://", "https://")), f"NOMIKOS_API_URL must be http(s): {API_URL}")


def api(method: str, path: str, token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API_URL + path, data=data, method=method)  # noqa: S310 - scheme checked above
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - scheme checked above
        raw = resp.read()
        return json.loads(raw) if raw else None


def login(email: str, password: str) -> str | None:
    try:
        return api("POST", "/auth/login", body={"email": email, "password": password})[
            "access_token"
        ]
    except urllib.error.HTTPError:
        return None


def ensure_user(email: str, username: str, password: str) -> str:
    token = login(email, password)
    if token:
        return token
    api("POST", "/auth/register", body={"email": email, "username": username, "password": password})
    token = login(email, password)
    need(token, f"could not log in as {email}")
    return token


def items(payload):
    return payload["items"] if isinstance(payload, dict) and "items" in payload else payload


@dataclass
class Seed:
    project_id: str
    document_id: str
    part_id: str

    @property
    def editor_path(self) -> str:
        return f"/projects/{self.project_id}/documents/{self.document_id}/parts/{self.part_id}"

    @property
    def document_path(self) -> str:
        return f"/projects/{self.project_id}/documents/{self.document_id}"


def find_seed(token: str) -> Seed:
    project = next(p for p in items(api("GET", "/projects", token)) if p["name"] == SEED_PROJECT)
    docs = items(api("GET", f"/projects/{project['id']}/documents", token))
    doc = next(d for d in docs if d["name"].startswith(SEED_DOCUMENT))
    detail = api("GET", f"/projects/{project['id']}/documents/{doc['id']}", token)
    part = sorted(detail["parts"], key=lambda p: p.get("order", 0))[0]
    return Seed(project["id"], doc["id"], part["id"])


def part_base(seed: Seed) -> str:
    return f"/projects/{seed.project_id}/documents/{seed.document_id}/parts/{seed.part_id}"


def reset_seed_part(token: str, seed: Seed) -> None:
    """Drop the regions an earlier run drew (labelled 4v or unlabelled)."""
    for block in api("GET", part_base(seed) + "/layout", token)["blocks"]:
        if (block.get("label") or "") in ("", "4v"):
            api("DELETE", f"{part_base(seed)}/blocks/{block['id']}", token)


def line_ids(token: str, seed: Seed) -> set[str]:
    return {line["id"] for line in api("GET", part_base(seed) + "/layout", token)["lines"]}


def upload_image(token: str, path: str, filename: str, data: bytes) -> dict:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/webp\r\n\r\n"
    )
    body = head.encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API_URL + path, data=body, method="POST")  # noqa: S310 - scheme checked above
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
    req.add_header("authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - scheme checked above
        return json.loads(resp.read())


def ground_truth(line: dict) -> str | None:
    texts = [
        t["text"] for t in line["line_transcriptions"] if t["transcription_kind"] == "ground_truth"
    ]
    return texts[0] if texts else None


def seed_parts(token: str, seed: Seed) -> list[dict]:
    detail = api("GET", seed.document_path, token)
    return sorted(detail["parts"], key=lambda p: p.get("order", 0))


def first_line_text(token: str, seed: Seed, index: int) -> str:
    """The start of the first transcribed line on seeded page ``index``, which
    the public page must show."""
    part = seed_parts(token, seed)[index]
    layout = api("GET", f"{seed.document_path}/parts/{part['id']}/layout", token)
    texts = [ground_truth(line) for line in sorted(layout["lines"], key=lambda x: x["order"])]
    text = next(t for t in texts if t)
    return text.strip()[:12]


def copy_seed_document(dev_token: str, seed: Seed, token: str, project_id: str) -> str:
    """Copy the seeded document, its page images, lines and ground truth text,
    into ``project_id`` as the user behind ``token``. Real seeded data, read
    and written through the public API."""
    doc = api("POST", f"/projects/{project_id}/documents", token, {"name": SEED_DOCUMENT})
    doc_path = f"/projects/{project_id}/documents/{doc['id']}"
    for part in seed_parts(dev_token, seed):
        req = urllib.request.Request(API_URL + part["image_url"])  # noqa: S310 - scheme checked above
        req.add_header("authorization", f"Bearer {dev_token}")
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - scheme checked above
            data = resp.read()
        new = upload_image(token, doc_path + "/parts", f"page-{part['order'] + 1}.webp", data)
        layout = api("GET", f"{seed.document_path}/parts/{part['id']}/layout", dev_token)
        lines = [
            {
                "order": line["order"],
                "points": line["points"],
                "baseline": line["baseline"],
                "mask": line["mask"],
                "kind": line["kind"],
                "approved_text": ground_truth(line),
            }
            for line in layout["lines"]
        ]
        api("PUT", f"{doc_path}/parts/{new['id']}/lines", token, {"lines": lines})
    return doc["id"]


def reset_user_projects(token: str) -> None:
    for project in items(api("GET", "/projects", token)):
        if project["name"] == PROJECT_NAME:
            api("DELETE", f"/projects/{project['id']}", token)


def cancel_open_jobs(token: str) -> None:
    """Cancel the queued jobs an earlier run left, so the editor buttons are free."""
    for project in items(api("GET", "/projects", token)):
        for job in items(api("GET", f"/projects/{project['id']}/jobs", token)):
            if job["status"] in ("pending", "running", "waiting"):
                api("POST", f"/jobs/{job['id']}/cancel", token)
    with contextlib.suppress(urllib.error.HTTPError):
        api("DELETE", "/jobs/history", token)


def share_project(token: str, project_id: str, email: str) -> None:
    try:
        api("POST", f"/projects/{project_id}/share", token, {"email": email})
    except urllib.error.HTTPError as err:
        if err.code not in (400, 409):
            raise


LOCAL_HOSTS = ("localhost", "127.0.0.1")


def guard_target(allow_remote: bool) -> None:
    """Refuse a non-local stack: the cleanup deletes projects, regions, Segments
    and job history through the API, so a typo in a URL must not reach a real
    deployment. ``--allow-remote`` is the explicit override."""
    remote = [url for url in (BASE_URL, API_URL) if urlparse(url).hostname not in LOCAL_HOSTS]
    if remote and not allow_remote:
        raise SystemExit(
            f"Refusing to run against a non-local stack: {', '.join(remote)}. "
            "The script deletes data there; pass --allow-remote if that is intended."
        )
    print(f"Frontend: {BASE_URL}")
    print(f"API:      {API_URL}")


# --- Recording ---------------------------------------------------------------


@dataclass
class Recorder:
    page: Page
    captions: dict[str, str]
    t0: float = field(default_factory=time.monotonic)
    log: list[tuple[float, str]] = field(default_factory=list)
    _last: float = 0.0

    def now(self) -> float:
        return time.monotonic() - self.t0

    def caption(self, key: str, dwell: float = 0.3) -> None:
        text = self.captions[key]
        need(len(text) < 90, f"caption {key!r} is {len(text)} characters")
        if self.log:
            wait = MIN_HOLD - (self.now() - self._last)
            if wait > 0:
                time.sleep(wait)
        self.page.evaluate("t => window.__setCaption(t)", text)
        self._last = self.now()
        self.log.append((self._last, text))
        time.sleep(dwell)

    def hold(self, seconds: float = MIN_HOLD) -> None:
        time.sleep(seconds)

    def point(self, target: Locator | tuple[float, float]) -> tuple[float, float]:
        if isinstance(target, tuple):
            x, y = target
        else:
            target.wait_for(state="visible", timeout=30000)
            target.scroll_into_view_if_needed()
            box = target.bounding_box()
            need(box, "element has no box")
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=12)
        return x, y

    def click(self, target: Locator | tuple[float, float]) -> None:
        x, y = self.point(target)
        self.page.mouse.click(x, y)

    def drag(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        self.point(start)
        self.page.mouse.down()
        self.page.mouse.move(end[0], end[1], steps=15)
        self.page.mouse.up()

    def type(self, target: Locator, text: str) -> None:
        self.click(target)
        self.page.keyboard.type(text, delay=25)

    def goto(self, path: str) -> None:
        self.page.goto(BASE_URL + path)


def new_context(browser, video_dir: Path):
    ctx = browser.new_context(
        viewport=VIEWPORT,
        record_video_dir=str(video_dir),
        record_video_size=VIEWPORT,
        accept_downloads=True,
    )
    ctx.add_init_script(INIT_SCRIPT)
    page = ctx.new_page()
    page.on("dialog", lambda dialog: dialog.accept())
    return ctx, page


def warm_up(browser, seed: Seed) -> None:
    """Visit every route once off camera, so a dev server compiles them now
    rather than in the middle of the recording."""
    ctx = browser.new_context(viewport=VIEWPORT)
    page = ctx.new_page()
    page.goto(BASE_URL + "/register")
    page.goto(BASE_URL + "/login")
    page.locator("input[type=email]").fill(DEV_EMAIL)
    page.locator("input[type=password]").fill(DEV_PASSWORD)
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/projects", timeout=60000)
    zero = "00000000-0000-0000-0000-000000000000"
    for path in (
        f"/projects/{seed.project_id}",
        seed.document_path,
        seed.editor_path,
        f"/public/projects/{zero}/documents/{zero}?t=warm-up",
    ):
        page.goto(BASE_URL + path)
        page.wait_for_load_state("load")
        page.wait_for_timeout(1000)
    ctx.close()


def ui_login(r: Recorder, email: str, password: str, caption: str | None = None) -> None:
    r.goto("/login")
    page = r.page
    page.locator("input[type=email]").wait_for()
    if caption:
        r.caption(caption)
    r.type(page.locator("input[type=email]"), email)
    r.type(page.locator("input[type=password]"), password)
    r.click(page.locator("button[type=submit]"))
    page.wait_for_url("**/projects", timeout=30000)
    page.get_by_role("button", name="New project").wait_for()


def dismiss_note(page: Page) -> None:
    note = page.get_by_role("button", name="Dismiss this note")
    if note.count() and note.first.is_visible():
        note.first.click()


def wait_editor(page: Page) -> None:
    page.locator("polygon[aria-label^='Segment 1,']").wait_for(timeout=60000)
    page.wait_for_timeout(800)
    dismiss_note(page)


def page_image_box(page: Page) -> dict:
    """The box around every Segment on the canvas, in viewport pixels."""
    boxes = page.locator("polygon[aria-label^='Segment ']").evaluate_all(
        "els => els.map(e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; })"
    )
    return {
        "x0": min(b[0] for b in boxes),
        "top": min(b[1] for b in boxes),
        "x1": max(b[2] for b in boxes),
        "bottom": max(b[3] for b in boxes),
    }


def fit_view(page: Page) -> None:
    page.get_by_role("button", name="Fit page to view").click()
    page.wait_for_timeout(500)


def toggle_transcript(r: Recorder) -> None:
    """The Transcript button in the top bar (its title says Hide or Show)."""
    r.click(r.page.locator("button[title$='transcript pane (T)']").first)


def draw_region(r: Recorder) -> None:
    page = r.page
    box = page_image_box(page)
    r.click(page.get_by_role("button", name="Region (G)"))
    pad = 18
    r.drag((box["x0"] - pad, box["top"] - pad), (box["x1"] + pad, box["bottom"] + pad))
    label = page.get_by_label("Region label")
    label.wait_for(timeout=10000)
    r.type(label, "4v")
    page.keyboard.press("Enter")
    r.hold(0.8)


def download(r: Recorder, target: Locator) -> None:
    """Start a download and move on; the file builds while the video goes on."""
    r.page.once("download", lambda d: d.cancel())
    r.click(target)


# --- The tour ------------------------------------------------------------------


def record_tour(
    browser, captions: dict[str, str], work: Path
) -> tuple[list[Path], list[tuple[float, str]], list]:
    token = login(DEV_EMAIL, DEV_PASSWORD)
    need(token, "dev login failed")
    seed = find_seed(token)
    reset_seed_part(token, seed)
    cancel_open_jobs(token)
    lines_before = line_ids(token, seed)
    warm_up(browser, seed)

    ctx, page = new_context(browser, work / "tour")
    r = Recorder(page, captions)
    r.goto("/login")
    page.locator("input[type=email]").fill(DEV_EMAIL)
    page.locator("input[type=password]").fill(DEV_PASSWORD)
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/projects")
    r.goto(seed.editor_path)
    wait_editor(page)
    fit_view(page)

    r.caption("pickers")
    page.get_by_label("HTR transcription model").select_option(label="greek-v1")
    r.point(page.get_by_label("HTR transcription model"))
    page.get_by_label("Segmentation model").select_option(label="seg-v1")
    r.point(page.get_by_label("Segmentation model"))

    r.caption("numbering")
    box = page_image_box(page)
    r.click(page.get_by_role("button", name="Rectangle segment (B)"))
    count_before = page.locator("polygon[aria-label^='Segment ']").count()
    y = box["top"] - 30
    x0, x1 = box["x0"] + 40, box["x0"] + 220
    r.drag((x0, y), (x1, y + 22))
    page.wait_for_function(
        f"document.querySelectorAll(\"polygon[aria-label^='Segment ']\").length > {count_before}",
        timeout=15000,
    )
    r.hold(1.2)
    r.click(page.get_by_role("button", name="Select and pan (Esc)"))
    r.click(((x0 + x1) / 2, y + 11))
    r.click(page.get_by_role("button", name="Delete selected (⌫)"))
    page.wait_for_function(
        f"document.querySelectorAll(\"polygon[aria-label^='Segment ']\").length === {count_before}",
        timeout=15000,
    )

    r.caption("pane")
    toggle_transcript(r)
    r.hold(1.0)
    toggle_transcript(r)

    r.caption("region")
    fit_view(r.page)
    draw_region(r)

    r.caption("workflow")
    r.click(page.get_by_role("button", name="Workflow"))
    r.point(page.get_by_role("menuitem", name="Check page", exact=False).first)
    r.point(page.locator("[aria-label='Download Word document']"))
    r.hold(0.8)
    page.keyboard.press("Escape")

    r.caption("jobs")
    r.click(page.get_by_role("button", name="Transcribe").first)
    pill = page.get_by_role("button", name="background job", exact=False)
    pill.wait_for(timeout=20000)
    r.click(pill)
    page.get_by_role("dialog", name="Background jobs").wait_for(timeout=20000)
    r.hold(3.0)

    r.caption("feedback")
    r.point(page.get_by_role("button", name="Send feedback"))
    r.hold(MIN_HOLD)

    video = page.video
    ctx.close()
    # The numbering demo deletes its Segment on screen, but a run that stopped
    # half way must not leave it in the seeded page; the job is cancelled too.
    for line_id in line_ids(token, seed) - lines_before:
        api("DELETE", f"{part_base(seed)}/lines/{line_id}", token)
    cancel_open_jobs(token)
    lead = max(r.log[0][0] - 0.3, 0)
    log = [(t - lead, c) for t, c in r.log]
    return [Path(video.path())], log, [(lead, None, 0)]


# --- The explainer ----------------------------------------------------------


def record_explainer(browser, captions: dict[str, str], work: Path):
    dev_token = login(DEV_EMAIL, DEV_PASSWORD)
    need(dev_token, "dev login failed")
    seed = find_seed(dev_token)
    cancel_open_jobs(dev_token)
    ensure_user(SECOND_EMAIL, SECOND_NAME, SECOND_PASSWORD)
    user_token = login(USER_EMAIL, USER_PASSWORD)
    if user_token:
        cancel_open_jobs(user_token)
        reset_user_projects(user_token)
    warm_up(browser, seed)

    ctx, page = new_context(browser, work / "explainer")
    r = Recorder(page, captions)

    # 1. Register, or log in when the account exists from an earlier run.
    if user_token is None:
        r.goto("/register")
        page.locator("#email").wait_for()
        r.caption("register")
        r.type(page.locator("#email"), USER_EMAIL)
        r.type(page.locator("#username"), USER_NAME)
        r.type(page.locator("#password"), USER_PASSWORD)
        r.click(page.get_by_role("button", name="Create account"))
        page.wait_for_url("**/projects", timeout=30000)
        page.get_by_role("button", name="New project").wait_for()
        user_token = login(USER_EMAIL, USER_PASSWORD)
        need(user_token, "the new user cannot log in")
    else:
        ui_login(r, USER_EMAIL, USER_PASSWORD, caption="login")

    # 2. Project and document.
    r.caption("project")
    r.click(page.get_by_role("button", name="New project"))
    r.type(page.locator("dialog input, [role=dialog] input").first, PROJECT_NAME)
    r.click(page.get_by_role("button", name="Create"))
    page.get_by_text(PROJECT_NAME, exact=True).first.wait_for(timeout=30000)
    if not page.get_by_role("button", name="New document").is_visible():
        r.click(page.get_by_role("button", name="Open").nth(_row_index(page, PROJECT_NAME)))
    page.get_by_role("button", name="New document").wait_for(timeout=30000)
    project_url = page.url
    project_id = next(
        p["id"] for p in items(api("GET", "/projects", user_token)) if p["name"] == PROJECT_NAME
    )
    # The narrator's transcribed document: the seeded Grec 1360 pages with their
    # ground truth, copied into her project while the camera shows the upload.
    pool = ThreadPoolExecutor(max_workers=1)
    copied = pool.submit(copy_seed_document, dev_token, seed, user_token, project_id)
    r.click(page.get_by_role("button", name="New document"))
    r.type(page.locator("dialog input, [role=dialog] input").first, DOCUMENT_NAME)
    r.click(page.get_by_role("button", name="Create"))
    page.get_by_text(DOCUMENT_NAME, exact=True).first.wait_for(timeout=30000)
    if not page.get_by_role("button", name="Upload pages").is_visible():
        r.click(page.get_by_role("button", name="Open").nth(_row_index(page, DOCUMENT_NAME)))
    page.get_by_role("button", name="Upload pages").wait_for(timeout=30000)

    # 3. Upload.
    r.caption("upload")
    with page.expect_file_chooser() as chooser:
        r.click(page.get_by_role("button", name="Upload pages"))
    chooser.value.set_files([str(IMAGE_DIR / name) for name in IMAGES])
    page.locator(f"[aria-label='Open part {len(IMAGES)} in editor']").wait_for(timeout=120000)
    r.hold(0.8)

    # 4. A segmentation job. No worker runs locally, so it stays queued.
    r.caption("segment")
    r.click(page.get_by_role("button", name="Workflow"))
    picker = page.get_by_label("Segmentation model")
    picker.select_option(label="seg-v1")
    r.point(picker)
    r.click(page.get_by_text("Segment unsegmented pages").first)
    page.wait_for_timeout(800)
    page.keyboard.press("Escape")
    r.click(page.get_by_role("link", name=PROJECT_NAME))
    queued = page.get_by_text("Segmentation", exact=True).first
    queued.wait_for(timeout=30000)
    r.caption("queued", dwell=0.2)
    r.point(queued)

    # 5. The editor, on her transcribed copy of Grec 1360.
    copied.result(timeout=120)
    pool.shutdown()
    page.reload()
    page.get_by_text(SEED_DOCUMENT, exact=True).first.wait_for(timeout=30000)
    r.caption("open_doc")
    r.click(page.get_by_role("button", name="Open").nth(_row_index(page, SEED_DOCUMENT)))
    page.locator("[aria-label='Open part 1 in editor']").wait_for()
    doc_url = page.url
    r.click(page.locator("[aria-label='Open part 1 in editor']"))
    wait_editor(page)
    fit_view(page)

    r.caption("select_line")
    r.click(page.locator("polygon[aria-label^='Segment 5,']"))
    r.caption("edit_line")
    text_box = page.get_by_label("Segment 5 text")
    r.click(text_box)
    page.keyboard.press("End")
    last = text_box.input_value()[-1:]
    page.keyboard.press("Backspace")
    r.hold(0.5)
    page.keyboard.type(last, delay=150)
    page.keyboard.press("Escape")

    r.caption("pane")
    toggle_transcript(r)
    r.hold(1.0)
    toggle_transcript(r)

    r.caption("region")
    fit_view(r.page)
    draw_region(r)
    r.click(page.get_by_role("button", name="Select and pan (Esc)"))

    # 6. One export.
    r.caption("export")
    r.click(page.get_by_role("button", name="Workflow"))
    word = page.locator("[aria-label='Download Word document']")
    r.point(word)
    r.hold(1.2)
    download(r, word)
    r.hold(0.8)

    # 7. Sharing the project with a colleague.
    r.goto(project_url.removeprefix(BASE_URL))
    page.get_by_role("button", name="Share", exact=True).wait_for()
    r.caption("share")
    r.click(page.get_by_role("button", name="Share", exact=True))
    r.type(page.get_by_placeholder("Email or username"), SECOND_EMAIL)
    r.click(page.locator("form.entity-panel__share-form button[type=submit]"))
    people = page.get_by_label("Collaborators")
    people.wait_for(timeout=15000)
    r.point(people)
    r.hold(1.0)

    # 8. Publishing the transcribed document and opening the public page.
    r.goto(doc_url.removeprefix(BASE_URL))
    page.get_by_role("button", name="Publish").wait_for()
    r.caption("publish")
    r.click(page.get_by_role("button", name="Publish"))
    r.click(page.get_by_text("Publish document (").first)
    r.click(page.get_by_text("Publish document", exact=True).last)
    link_box = page.get_by_label("Public document link")
    link_box.wait_for(timeout=20000)
    r.point(link_box)
    r.hold(0.8)
    public_url = link_box.input_value()
    split_at = r.now()

    ctx2, page2 = new_context(browser, work / "explainer-public")
    r2 = Recorder(page2, captions)
    page2.goto(public_url)
    page2.get_by_text(SEED_DOCUMENT).first.wait_for(timeout=60000)
    page2.get_by_text(first_line_text(dev_token, seed, 0)).first.wait_for(timeout=60000)
    page2.wait_for_timeout(500)
    r2.caption("public", dwell=1.0)
    r2.click(page2.get_by_text("Page 2", exact=True).first)
    page2.get_by_text(first_line_text(dev_token, seed, 1)).first.wait_for(timeout=60000)
    r2.caption("public_page2", dwell=MIN_HOLD)
    public_start = max(r2.log[0][0] - 0.8, 0)
    public_end = r2.now()
    video2 = page2.video
    ctx2.close()
    video1 = page.video
    ctx.close()

    # The final cut: this context up to the public link, then the fresh
    # context's public page. The lead-in before the first caption is trimmed.
    lead = max(r.log[0][0] - 0.3, 0)
    first = split_at - lead
    log = [(t - lead, c) for t, c in r.log if t <= split_at]
    log += [(first + t - public_start, c) for t, c in r2.log]
    segments = [(lead, split_at, 0), (public_start, public_end, 1)]
    return [Path(video1.path()), Path(video2.path())], log, segments


def _row_index(page: Page, name: str) -> int:
    """Which Open button belongs to the row whose text starts with ``name``."""
    rows = page.get_by_role("button", name="Open")
    for i in range(rows.count()):
        row_text = rows.nth(i).evaluate(
            "b => (b.closest('tr, li, article, [role=row], .entity-row') || b.parentElement.parentElement).innerText"
        )
        if name in row_text:
            return i
    raise AssertionError(f"no row for {name!r}")


# --- Output ------------------------------------------------------------------


def encode(videos: list[Path], segments: list, out: Path) -> None:
    if len(videos) == 1:
        start = segments[0][0]
        cmd = [FFMPEG, "-y", "-i", str(videos[0])]
        cmd += ["-vf", f"trim=start={start:.3f},setpts=PTS-STARTPTS,fps=25"]
    else:
        cmd = [FFMPEG, "-y"]
        for video in videos:
            cmd += ["-i", str(video)]
        parts = []
        for i, (start, end, src) in enumerate(segments):
            trim = f"start={start:.3f}" + (f":end={end:.3f}" if end is not None else "")
            parts.append(f"[{src}:v]trim={trim},setpts=PTS-STARTPTS,fps=25[s{i}]")
        joined = "".join(f"[s{i}]" for i in range(len(segments)))
        graph = ";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0[out]"
        cmd += ["-filter_complex", graph, "-map", "[out]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603 - fixed ffmpeg argv, no shell


def write_captions(log: list[tuple[float, str]], out: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Edit the wording in scripts/platform/explainer/captions.json and re-record.",
        "",
    ]
    lines += ["| # | Time | Caption |", "|---|------|---------|"]
    for i, (t, text) in enumerate(log, 1):
        lines.append(f"| {i} | {int(t // 60)}:{t % 60:04.1f} | {text} |")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", choices=["tour", "explainer"], required=True)
    parser.add_argument("--headed", action="store_true", help="show the browser while recording")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="record against a stack that is not on localhost (the cleanup deletes data there)",
    )
    args = parser.parse_args()
    guard_target(args.allow_remote)

    captions = json.loads((HERE / "captions.json").read_text())[args.video]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = OUT_DIR / "raw"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed, slow_mo=SLOW_MO)
        try:
            if args.video == "tour":
                videos, log, segments = record_tour(browser, captions, work)
                mp4, md, title = (
                    "nomikos-whats-new.mp4",
                    "captions-whats-new.md",
                    "Nomikos: what is new",
                )
            else:
                videos, log, segments = record_explainer(browser, captions, work)
                mp4, md, title = "nomikos-explainer.mp4", "captions.md", "Nomikos explainer"
        except Exception:
            # Leave a picture of where it stopped; a moved button is the usual cause.
            for i, ctx in enumerate(browser.contexts):
                for j, page in enumerate(ctx.pages):
                    page.screenshot(path=str(OUT_DIR / f"failed-{i}-{j}.png"))
            raise
        browser.close()
    encode(videos, segments, OUT_DIR / mp4)
    write_captions(log, OUT_DIR / md, title)
    print(OUT_DIR / mp4)
    print(OUT_DIR / md)


if __name__ == "__main__":
    main()
