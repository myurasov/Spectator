# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Centralized constants & defaults for Spectator.

Source-of-truth for paths, image refs, profile names. Values that the user
needs to override (NGC keys, remote LLM endpoint, hardware profile) live
here as defaults but each CLI subcommand also accepts a flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

VSS_REPO_URL = "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git"
VSS_VERSION = "v3.1.0"

DEFAULT_REMOTE_WORKDIR = "~/spectator"
DEFAULT_VSS_CHECKOUT = "video-search-and-summarization"

DEFAULT_HARDWARE_PROFILE = "DGX-SPARK"
DEFAULT_DEPLOY_PROFILE = "base"
DEFAULT_REMOTE_LLM = "nvidia/nvidia-nemotron-nano-9b-v2"
DEFAULT_LLM_ENDPOINT = "https://integrate.api.nvidia.com/v1"

# VSS v3.1 host ports (`base` workflow on a self-hosted Spark):
#   UI_PORT        (3030)  — vss-ui Next.js. Upstream compose hardcodes 3000:3000;
#                            Spectator's install step patches the host side to
#                            ${VSS_UI_PORT:-3030} so port 3000 stays free for
#                            other user tooling (whatsapp bridges, dev servers, etc).
#   AGENT_API_PORT (8000)  — vss-agent REST API.
#   PROXY_PORT     (7777)  — vss-proxy. Not deployed in the on-prem `base` workflow;
#                            only the cloud-Brev profile (`*_proxy`) activates it.
UI_PORT = 3030
AGENT_API_PORT = 8000
PROXY_PORT = 7777

# User-local cache cleaner path (vs. the playbook's /usr/local/bin/ which would
# need sudo to write). The script itself still needs sudo at runtime because it
# writes to /proc/sys/vm/{nr_hugepages,drop_caches}, but the script *file*
# stays in user space — no global filesystem pollution.
CACHE_CLEANER_RELPATH = "bin/sys-cache-cleaner.sh"  # under workdir

REQUIRED_DRIVER = "580.95.05"
REQUIRED_CUDA_MAJOR_MINOR = "13.0"
REQUIRED_DOCKER = "27.2.0"
REQUIRED_COMPOSE = "v2.29.0"


@dataclass(slots=True)
class StackConfig:
    """Resolved deployment configuration for one VSS bring-up.

    Built from CLI flags + env. Anything not provided falls back to the
    values above. CLI subcommands receive an instance of this and pass it
    to the install / stack / api modules.
    """

    workdir: str = DEFAULT_REMOTE_WORKDIR
    vss_checkout: str = DEFAULT_VSS_CHECKOUT
    hardware_profile: str = DEFAULT_HARDWARE_PROFILE
    deploy_profile: str = DEFAULT_DEPLOY_PROFILE
    remote_llm: str = DEFAULT_REMOTE_LLM
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT
    ngc_api_key: str | None = None
    nvidia_api_key: str | None = None

    def env_block(self) -> dict[str, str]:
        """Environment variables the dev-profile.sh script expects.

        Empty values are dropped (so the remote shell doesn't inherit
        empty strings that pass `[ -n ... ]` checks)."""
        env = {
            "NGC_CLI_API_KEY": self.ngc_api_key or "",
            "LLM_ENDPOINT_URL": self.llm_endpoint,
            "NVIDIA_API_KEY": self.nvidia_api_key or "",
        }
        return {k: v for k, v in env.items() if v}

    @classmethod
    def from_env(cls, **overrides) -> "StackConfig":
        """Build a StackConfig from process environment + explicit overrides.

        Spectator deliberately does not reach outside its own folder: it
        looks at $NGC_CLI_API_KEY / $NVIDIA_API_KEY in the current process
        env and nothing else. The user is responsible for getting those
        into the env (typically by `source`-ing a creds file from the
        shell that invokes Spectator). Typer's `envvar=` on each CLI flag
        is the canonical pickup path.
        """
        kw: dict[str, object] = {
            "ngc_api_key": os.environ.get("NGC_CLI_API_KEY"),
            "nvidia_api_key": os.environ.get("NVIDIA_API_KEY"),
        }
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)
