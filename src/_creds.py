# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Credential persistence at ``$workdir/.creds``.

Read priority (highest first):
  1. ``$workdir/.creds`` — sourced by every bash payload that needs creds.
  2. SSH-propagated env / process env — set by ``--ngc-key`` / ``--nvidia-key``
     / shell ``export``.
  3. None — caller errors if a required value is empty.

This ordering is deliberately backwards from the usual ``CLI > env > config``:
``.creds`` is the persistent source of truth that a user sets up once per
target. CLI flags / env vars are still useful for the **first** install
(they're what gets captured into ``.creds``), but after that the file is
authoritative. Rotating keys means editing the file directly.

Write: ``spectator install`` creates ``$workdir/.creds`` the first time,
capturing whatever Spectator-managed env vars are set in the install shell.
Subsequent installs leave the file alone.

Format: shell-source-able. Each line is ``export VAR=VALUE`` with the
value shell-quoted via ``printf %q``. ``chmod 600`` so other users on the
host can't read it. The ``.creds`` filename is in deploy.py's rsync
excludes — secrets never travel over rsync, only the SSH env or
direct-write paths.
"""

from __future__ import annotations

import textwrap

CREDS_RELPATH = ".creds"

# Env vars Spectator persists into ``.creds``. Order matters only for the
# rendered file (lines appear in this order). Adding a new var here is
# enough to make it round-trip through .creds.
#
# - NGC_CLI_API_KEY / NVIDIA_API_KEY / LLM_ENDPOINT_URL: VSS Blueprint
#   bring-up + remote-LLM hookup.
# - HUGGING_FACE_HUB_TOKEN: pyannote.audio model downloads (used by
#   ``spectator audio diarize`` and ``audio transcribe --diarize``).
#   This is the canonical name; the diarize CLI also accepts the
#   shorter ``HF_TOKEN`` alias for convenience but persists under
#   the canonical name only.
CREDS_VARS: tuple[str, ...] = (
    "NGC_CLI_API_KEY",
    "NVIDIA_API_KEY",
    "LLM_ENDPOINT_URL",
    "HUGGING_FACE_HUB_TOKEN",
)


def source_block(workdir_bash: str) -> str:
    """Return a bash one-liner that sources ``$workdir_bash/.creds`` if it
    exists. Designed for inclusion at the TOP of any bash payload that
    needs creds.

    ``workdir_bash`` should be a bash-friendly form (typically
    ``$HOME/.spectator`` via :func:`audio._expand_tilde`) — a literal
    ``~/...`` inside double quotes does not tilde-expand.

    ``set -a`` makes any plain ``VAR=VALUE`` line export-equivalent so
    .creds files written without ``export`` (e.g. hand-edited) still
    work as expected. The ``|| true`` swallows a non-zero exit from
    sourcing (e.g. parse error in a hand-edited .creds) so the
    surrounding ``set -e`` doesn't abort the whole payload before the
    user has a chance to see other diagnostics.
    """
    path = f"{workdir_bash}/{CREDS_RELPATH}"
    return f'[ -f "{path}" ] && {{ set -a; . "{path}"; set +a; }} || true'


def save_block(workdir_bash: str) -> str:
    """Return a bash block that writes ``$workdir_bash/.creds`` if it
    doesn't already exist, capturing the current values of every var
    in :data:`CREDS_VARS`.

    Empty / unset vars are skipped (the file might end up empty if
    nothing's set; the source_block tolerates that). ``printf %q``
    shell-quotes each value safely. ``chmod 600`` keeps the file
    readable only by the owner.
    """
    path = f"{workdir_bash}/{CREDS_RELPATH}"
    var_lines = []
    for v in CREDS_VARS:
        var_lines.append(textwrap.dedent(f'''\
              if [ -n "${{{v}:-}}" ]; then
                printf "export {v}=%s\\n" "$(printf %q "${{{v}}}")" >> "$CREDS_FILE"
              fi'''))
    var_block = "\n".join(var_lines)
    return textwrap.dedent(f'''
        CREDS_FILE="{path}"
        if [ ! -f "$CREDS_FILE" ]; then
          printf "# Spectator credentials.\\n" > "$CREDS_FILE"
          printf "# Sourced by every bash payload that needs creds.\\n" >> "$CREDS_FILE"
          printf "# Edit this file directly to rotate keys.\\n" >> "$CREDS_FILE"
          printf "# Generated %s by spectator install.\\n" "$(date)" >> "$CREDS_FILE"
{var_block}
          chmod 600 "$CREDS_FILE"
          echo "==== wrote $CREDS_FILE (creds persisted; future calls source it) ===="
        fi
    ''').strip()


def ensure_var_block(workdir_bash: str, var: str) -> str:
    """Bash block that idempotently ensures ``$var`` is present in
    ``$workdir/.creds`` when it's set in the current environment.

    Three cases handled:

    1. ``.creds`` doesn't exist + ``$var`` is set → creates the file
       (header + ``export $var=<quoted value>``, ``chmod 600``).
    2. ``.creds`` exists but doesn't contain a line matching
       ``^export $var=`` + ``$var`` is set → appends one line.
    3. ``$var`` is empty / unset, or already present in ``.creds`` →
       no-op.

    Designed for late-arriving credentials like
    ``HUGGING_FACE_HUB_TOKEN`` that a feature added after first
    install ((audio install ran before the user had an HF account).
    Pairs with :func:`source_block` and :func:`save_block` —
    ``save_block`` is the first-install path (writes the file with
    everything in :data:`CREDS_VARS` in one shot);
    ``ensure_var_block`` is the per-var late-arrival path that
    survives the "file already exists but is missing this one var"
    case.
    """
    if var not in CREDS_VARS:
        raise ValueError(f"{var!r} not in CREDS_VARS; add it there first")
    path = f"{workdir_bash}/{CREDS_RELPATH}"
    return textwrap.dedent(f'''
        if [ -n "${{{var}:-}}" ]; then
          CREDS_FILE="{path}"
          if [ ! -f "$CREDS_FILE" ]; then
            printf "# Spectator credentials.\\n# Generated %s by spectator.\\n" "$(date)" > "$CREDS_FILE"
            chmod 600 "$CREDS_FILE"
          fi
          if ! grep -q "^export {var}=" "$CREDS_FILE" 2>/dev/null; then
            printf "export {var}=%s\\n" "$(printf %q "${{{var}}}")" >> "$CREDS_FILE"
            echo "==== persisted {var} to $CREDS_FILE ===="
          fi
        fi
    ''').strip()


__all__ = ["CREDS_RELPATH", "CREDS_VARS", "ensure_var_block", "save_block", "source_block"]
