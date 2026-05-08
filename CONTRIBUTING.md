# Contributing to Spectator

Thanks for your interest in contributing! Spectator is a personal open-source project, released under the [Apache License 2.0](LICENSE). This document explains the contribution workflow, the developer-certificate-of-origin (DCO) sign-off requirement, and the basic coding conventions.

## Table of Contents

- [Issue Tracking](#issue-tracking)
- [Pull Requests](#pull-requests)
- [Coding Guidelines](#coding-guidelines)
- [Signing Your Work (DCO)](#signing-your-work-dco)
- [Reporting Security Issues](#reporting-security-issues)

## Issue Tracking

- All enhancement, bug-fix, or change requests should start with a [GitHub Issue](https://github.com/myurasov/Spectator/issues) so the design / scope can be discussed before code is written.
- For security-sensitive issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## Pull Requests

The developer workflow:

1. **Fork** the upstream repository: https://github.com/myurasov/Spectator
2. **Branch** from `main` in your fork; one branch per logical change.
3. **Develop** locally:

   ```bash
   # rebuild the venv from a clean state
   ./spectator install --force

   # pytest
   ./spectator test

   # ruff check
   ./spectator lint

   # ruff check --fix + ruff format
   ./spectator fmt
   ```

4. **Sign off** every commit with `git commit -s` (see [Signing Your Work](#signing-your-work-dco)). Unsigned commits will not be accepted.
5. **Open a PR** from your fork's branch into `main` of the upstream repo. Use a descriptive title in the imperative mood (e.g. `Fix preflight CUDA detection on driverless hosts`, not `fixed preflight`).
6. **Reference the issue number** in the PR body if there's a corresponding issue (e.g. `Closes #42`).
7. While under review, prefix work-in-progress PRs with `[WIP]`.

A reviewer will look at the PR. Please respond to review comments promptly; PRs that go silent for > 30 days may be closed (and can always be reopened).

## Coding Guidelines

- Follow the existing style in the file you're editing. Spectator uses `ruff` for both linting and formatting; running `./spectator fmt` will auto-fix most issues.
- Internal package imports use **relative form** (`from . import config`, `from ._run import ssh_run`) — keeps the package portable if the import name ever changes.
- Subprocess / SSH calls go through `_run.run` / `_run.ssh_run` / `_run.ssh_stream`. Don't shell out via raw `subprocess.run` from new call-sites — adding a primitive to `_run.py` is the right factoring.
- The default install path **never** writes outside `$workdir` and `~/.docker/config.json`. System-level mutations (`nvidia-ctk runtime configure`, docker group, `systemctl restart docker`) live behind the `--apply-system` flag. New code that touches system state outside `--apply-system` requires reviewer sign-off.
- Every new source file must include the SPDX license header (see existing files for the exact format):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <year> <Your Name>
  # SPDX-License-Identifier: Apache-2.0
  ```

  When making a substantial contribution to an existing file, you may add your copyright on a new line below the existing one — multiple copyright holders per file are fine.

- Keep PRs focused. If you find unrelated bugs while working on something, file a separate issue / PR.
- Update tests under `tests/` whenever you add or change behavior. The smoke test (`tests/test_smoke.py`) catches import-level regressions; behavior tests are welcome additions.
- Update `README.md` and/or `REFERENCE.md` whenever the user-facing surface changes.
- Update [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) in the same PR whenever you add, remove, or upgrade a runtime / dev dependency in `pyproject.toml`.
- Commit messages should be in imperative mood (`Add foo`, `Fix bar`, `Refactor baz`) and keep the subject ≤ 72 chars when practical. A short body explaining *why* is helpful for non-trivial changes.

## Signing Your Work (DCO)

I require every contributor to sign off on their commits. The sign-off certifies that the contribution is your original work, or you have rights to submit it under the project's license. This is the standard [Developer Certificate of Origin (DCO)](https://developercertificate.org/) — the same one used by the Linux kernel, Docker, and many other open-source projects.

**To sign off on a commit**, use the `--signoff` (or `-s`) flag:

```bash
git commit -s -m "Add cool feature"
```

This appends a line to the commit message:

```
Signed-off-by: Your Name <your@email.com>
```

`Your Name` and `your@email.com` must match your `git config user.name` and `git config user.email`. Anonymous contributions (no real name, no real email) cannot be accepted.

PRs containing unsigned commits will be blocked until every commit is signed. To retroactively sign existing commits, use `git rebase --signoff <base>` (or `git commit --amend --signoff` for the most recent one).

### Full text of the DCO

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Reporting Security Issues

Please do **not** report security vulnerabilities through public GitHub issues. See [SECURITY.md](SECURITY.md) for the responsible-disclosure process.
