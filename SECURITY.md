# Security Policy

If you find a security vulnerability in Spectator, please report it privately rather than opening a public GitHub issue. I appreciate the time you've taken to find it and will respond as quickly as I can.

## How to report

Use one of these private channels:

- **GitHub Security Advisory**: open a [private vulnerability report](https://github.com/myurasov/Spectator/security/advisories/new) — this is the preferred path; it gives us a private workspace to coordinate the fix and a CVE if appropriate.
- **Email**: send a short note to the maintainer (see the `authors` field of [`pyproject.toml`](pyproject.toml) for the current contact). Encrypt with a public PGP key if you have a sensitive proof-of-concept.

In your report, please include:

1. The Spectator version / branch / commit that reproduces the issue.
2. The type of vulnerability (e.g. command injection, path traversal, unauthenticated remote code execution).
3. Step-by-step instructions to reproduce.
4. A proof-of-concept or exploit script if you have one.
5. The potential impact, in your own words — who can exploit it, and what they get.

## Scope

Spectator is a thin wrapper around upstream NVIDIA Video Search & Summarization, OpenAI Whisper, and PyTorch. Vulnerabilities in those upstream projects are out of scope here — please report them upstream:

- VSS Blueprint: https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
- Whisper: https://github.com/openai/whisper
- PyTorch: https://github.com/pytorch/pytorch
- NVIDIA Product Security (NIM endpoints, NGC, drivers, container toolkit): https://www.nvidia.com/en-us/security/

If you're not sure which project the bug lives in, send the report to me first and I'll route it.

## What to expect

- I will acknowledge your report within 5 business days.
- I will work with you to confirm the issue and agree on a disclosure timeline. Default is 90 days from acknowledgement, but I'll negotiate something shorter or longer if the situation warrants.
- Once a fix is published, I will credit you in the release notes (unless you prefer to stay anonymous).

Spectator does not run a paid bug bounty.
