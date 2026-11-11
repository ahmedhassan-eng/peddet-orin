# Rules for changes to this repository

These rules apply to everyone who changes this repository, whether a person or an automated agent.

- This repository uses no commit trailers.
- Every published number comes from a JSON log under `runs/`; regenerate the README table with `python scripts/generate_table.py --write` and never edit it by hand.
- Run `ruff check .` and `python -m pytest` before opening a pull request.

The commit-message rule below is shared by this project's repositories and is copied here byte for byte. `tests/test_agents_rule.py` fails if the text between the markers changes.

<!-- shared-rules v2 sha256=1900994bbf2c4f6c7090742d7457c66e7988fd0293d6b91023924c3cca987686 -->
<!-- BEGIN SHARED RULES v2 -->
## Commit messages

Subject: imperative, no trailing period, aim for 50-72 characters.
Shorter is right when the change is genuinely small -- never pad a subject
to reach 50. When rewriting an existing message, lengthen its subject only
from substance that message already states: never from the diff, never
from inference.

Body: bullets only, imperative, one claim each -- two lines or fewer where
the claim allows. A bullet carrying several claims is prose with a marker.
As many bullets as there are claims, one per bullet. Six is guidance,
not a cap: fewer when the message has less to say, none when the old
message has no body, more whenever there are more claims.

Bullets are imperative when they state what the change does, and plain
indicative when they record a fact or a past defect. Both are statements.
Never pad a fact into an imperative.

Each bullet says what changed and why, scoped to the file or component
where that helps. Keep every number, ratio, commit id, file path, symbol
name and quoted string the change turns on.

A fact must be PRESENT in the new message. Repetition need not be
preserved: a reduction from two mentions to one is reported, not repaired.
An absence is a failure.

Leave out command output, gate transcripts and session narration. An error
contract the change is about may be quoted inline.

Trailers: exactly the ones this repository already uses.

Applies to commits made from now on. History is rewritten only by operator
ruling.

### Precedence, highest first

1. Invent nothing -- no claim, number or rationale the old message does not
   assert.
2. Preserve every claim -- a claim is dropped only by explicit decision,
   named in the report.
3. No fragments -- a bullet is a statement.
4. Length and bullet count -- the 72 ceiling, the 50 target and the
   six-bullet guidance yield to all three above. A message with eight
   claims gets eight bullets.

So a subject stays over 72 when shortening would drop a claim or leave a
fragment, and a message exceeds six bullets when it has more than six
claims. Every such case is listed in the report with its reason, and
every message over six bullets with its claim count.
<!-- END SHARED RULES v2 -->
