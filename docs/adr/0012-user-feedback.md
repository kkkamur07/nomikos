# 0012. User feedback

- Status: Accepted
- Date: 2026-10-11

## Context

There was no channel from inside the platform back to the owner: a researcher
who hit something confusing had to leave the app and write an email, so most
small frictions went unreported. Whatever is built must cost almost no space
in the interface and almost no new machinery on the server.

## Decision

1. **One table, `feedback`** (migration `009_feedback`): the author
   (`user_id`, cascade on delete), the note (`message`, 1 to 2000 chars),
   and the app route the user was on (`page_path`, nullable). No status, no
   threads, no attachments: triage happens outside the app, and every extra
   column is a workflow nobody asked for.
2. **One write route, `POST /feedback`**, for any signed-in user. The message
   is trimmed server-side and refused with 422 when nothing (or too much) is
   left. Each user is capped at 20 notes an hour (a count query, 429 past
   that), so a stuck client cannot flood the table.
3. **One read route, `GET /admin/feedback?limit=50`**, behind the same
   `require_admin` guard as the overview: newest first with the author's email
   and username. The dashboard renders it as one compact table next to the
   other cards.
4. **No `feedback_count` on the overview.** The overview's headline numbers
   are user, job, and failure counts; a feedback count has no natural slot
   there, and the section carries its own rows. The overview is untouched.

## Consequences

- Feedback is write-once from the user's side: there is no edit or delete,
  so a typo stays. A short note is cheap to resend, which is the fix.
- Notes vanish with their author through the cascade, so deleting an account
  deletes what it wrote.
- Production never runs alembic automatically, so the table ships as
  `scripts/platform/sql/2026-10-11_feedback.sql`, applied before the API
  build that serves the two routes.
