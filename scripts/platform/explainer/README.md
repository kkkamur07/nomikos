# Nomikos videos

`record_explainer.py` records two silent videos with subtitles burned into the picture:

| `--video` | Output | Length | What it shows |
|---|---|---|---|
| `tour` | `nomikos-whats-new.mp4`, `captions-whats-new.md` | about 1:40 | What changed this round, on the seeded Grec 1360 document as the dev user |
| `explainer` | `nomikos-explainer.mp4`, `captions.md` | about 3:00 | A new user from sign up to export, plus sharing and the public page |

Captions live in `captions.json`. Edit the wording there and re-record. Each caption has to stay under 90
characters (the script checks) and is held for at least 2.5 seconds.

## Re-record

```sh
export NOMIKOS_BASE_URL=http://localhost:3160      # frontend
export NOMIKOS_API_URL=http://localhost:8160       # API
export NOMIKOS_VIDEO_OUT=/some/folder              # default: ./out next to the script (gitignore it or pick a scratch folder)
PY=/Users/krishuagarwal/.pyenv/versions/3.11.10/bin/python   # any Python with Playwright and Chromium
$PY scripts/platform/explainer/record_explainer.py --video tour
$PY scripts/platform/explainer/record_explainer.py --video explainer
```

The script refuses to run unless both URLs are on `localhost` or `127.0.0.1`, because its cleanup deletes
data through the API; pass `--allow-remote` to override, and it prints both URLs before the first request.

Add `--headed` to watch the browser. `NOMIKOS_SLOW_MO` (default 400 ms) sets the pace and `FFMPEG` the
ffmpeg binary (default `/opt/homebrew/bin/ffmpeg`).

Other variables, all optional:

| Variable | Default |
|---|---|
| `NOMIKOS_DEV_EMAIL`, `NOMIKOS_DEV_PASSWORD` | `dev@example.com`, `dev-pass-123` (owner of the seeded project) |
| `NOMIKOS_USER_EMAIL`, `NOMIKOS_USER_NAME`, `NOMIKOS_USER_PASSWORD` | Eleni Markou, the new user of the explainer |
| `NOMIKOS_SECOND_EMAIL`, `NOMIKOS_SECOND_NAME`, `NOMIKOS_SECOND_PASSWORD` | Nikos Papadakis, the colleague she shares with (created through the API if missing) |
| `NOMIKOS_SEED_PROJECT`, `NOMIKOS_SEED_DOCUMENT` | `Byzantine Greek Manuscripts`, `Grec 1360` (name prefix) |
| `NOMIKOS_IMAGE_DIR`, `NOMIKOS_IMAGES` | the annotated data pages folder, two Grec 1360 jpegs (comma separated) |

## What the stack needs

- Seeded with `seed_dev_nomikos.py` and `seed_dev_inference.py` (the pickers list the catalog rows).
- The dev user is an admin, for the tour's Admin page:
  `update users set is_admin = true where email = 'dev@example.com';`
- An inference host that is online. Without one the API refuses every job ("No inference host is available")
  and the job steps fail. A local stack has none, so the recordings used a stand-in cloud device that never
  claims anything; jobs then sit queued, which is what the captions say:

  ```sql
  insert into helper_devices (id, user_id, name, platform, helper_version, last_seen_at, inference_host)
  select gen_random_uuid(), id, 'video stand-in (never claims)', 'linux', '0', now() + interval '30 days', 'cloud'
  from users where email = 'dev@example.com';
  ```

  Never do this on a stack with a real worker: delete the row after recording.

## Re-runs

The script tidies up through the API before it records: it cancels open jobs, deletes regions labelled
`4v` (or unlabelled) on the first seeded page, deletes Eleni's "Athos Letters" project, and after the tour
deletes any Segment the numbering demo left behind. The page text edit types back the character it removes,
so the seeded transcript is unchanged.

The explainer registers Eleni through the Register page only when the account does not exist yet; on a
re-run it shows the login instead. For the Register version again, delete her first:
`delete from users where email = 'eleni.markou@example.org';`

Before recording, the script opens every route once off camera, so a `next dev` server compiles them
before the camera rolls. When a step fails (usually a button that moved or was renamed) the script saves
`failed-*.png` in the output folder showing where it stopped.

## How it works

- Playwright records at 1440x900. An init script adds a fixed caption bar (text kept in `sessionStorage`
  so it survives navigations), a red dot that follows the mouse, and hides the Next.js dev badge.
- The public page is opened in a second, fresh browser context. The final cut is assembled with ffmpeg:
  the main recording up to the public link, the fresh context, then the main recording from the closing
  caption. The uncaptioned lead-in is trimmed. Output is H.264, yuv420p, faststart, no audio.
