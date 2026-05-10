# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Spectator: a thin CLI on top of NVIDIA's Video Search & Summarization
(VSS) Blueprint v3.1.

Wraps the install / deploy / lifecycle steps from
https://build.nvidia.com/spark/vss/instructions and the upstream repo
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
so a single `spectator deploy --target <gpu-machine>` does the whole
thing on the Spark, and `spectator process video.mp4` ships a video
through the running stack to get a Q&A-ready summary.
"""

__version__ = "0.3.2"
