# Third-Party Notices

Spectator is licensed under the [Apache License 2.0](LICENSE). It depends on the third-party open-source software listed below. Each dependency is distributed under its own license; license texts are available in the linked upstream repositories and are pulled into the runtime environment via the standard Python packaging tooling (`uv` / `pip`).

## Runtime dependencies

These packages are required for Spectator to run. They are pulled at install time (`./spectator install` or `uv sync`); their source code is not redistributed within this repository.

| Package | License | Project URL |
|---|---|---|
| [typer](https://github.com/fastapi/typer) | MIT | https://github.com/fastapi/typer |
| [rich](https://github.com/Textualize/rich) | MIT | https://github.com/Textualize/rich |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | https://github.com/encode/httpx |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | https://github.com/yaml/pyyaml |
| [fastapi](https://github.com/fastapi/fastapi) (v0.2.0+) | MIT | https://github.com/fastapi/fastapi |
| [uvicorn](https://github.com/encode/uvicorn) (v0.2.0+) | BSD-3-Clause | https://github.com/encode/uvicorn |
| [python-multipart](https://github.com/Kludex/python-multipart) (v0.2.0+) | Apache-2.0 | https://github.com/Kludex/python-multipart |

## Development dependencies

These packages are used only for the project's own development workflow (`./spectator test`, `./spectator lint`, `./spectator fmt`); they are not required at runtime and are not part of the published package.

| Package | License | Project URL |
|---|---|---|
| [pytest](https://github.com/pytest-dev/pytest) | MIT | https://github.com/pytest-dev/pytest |
| [ruff](https://github.com/astral-sh/ruff) | MIT | https://github.com/astral-sh/ruff |

## External services & blueprints

Spectator orchestrates two NVIDIA components that it does **not** redistribute. They are downloaded / pulled separately by `./spectator install` and `./spectator audio install`:

| Component | License | Project URL |
|---|---|---|
| [NVIDIA Video Search & Summarization (VSS) Blueprint v3.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) | Apache-2.0 | https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT | https://github.com/openai/whisper |
| [PyTorch](https://pytorch.org/) (Whisper runtime) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| [NVIDIA NIM API endpoints](https://build.nvidia.com/) (remote LLM inference) | NVIDIA service terms | https://build.nvidia.com/ |

Container base images and Docker images pulled from `nvcr.io` during VSS bring-up are governed by their own license terms; consult `nvcr.io` for details.

## License-Included Software

This project will download and install additional third-party open-source software projects (VSS Blueprint, Whisper, PyTorch, container base images). Review the license terms of these open-source projects before use.

## Updates

When dependencies are added, removed, or upgraded across major versions, update this file in the same commit. Run `./spectator install --force` after edits to `pyproject.toml` to ensure the lockfile is consistent.
