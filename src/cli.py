# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Spectator: CLI entrypoint (typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import (
    api,
    audio as audio_mod,
    config,
    deploy as deploy_mod,
    install as install_mod,
    preflight as preflight_mod,
    stack as stack_mod,
)

app = typer.Typer(
    name="spectator",
    help="Thin CLI on top of NVIDIA's Video Search & Summarization (VSS) Blueprint v3.1.\n\n"
         "Designed for DGX Spark (GB10) but works on any v3.1-supported host.",
    add_completion=False,
    no_args_is_help=True,
)
system_app = typer.Typer(
    name="system",
    help="System-level helpers (cache-cleaner start/stop, etc.). Each requires sudo.",
    no_args_is_help=True,
)
app.add_typer(system_app)

audio_app = typer.Typer(
    name="audio",
    help="Audio-only transcription via Whisper (sibling to the VSS video pipeline). "
         "Same containment policy: writes only into $workdir on the target.",
    no_args_is_help=True,
)
app.add_typer(audio_app)
console = Console()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_cfg(
    workdir: str | None,
    profile: str | None,
    hardware: str | None,
    remote_llm: str | None,
    llm_endpoint: str | None,
    ngc_key: str | None,
    nvidia_key: str | None,
) -> config.StackConfig:
    return config.StackConfig.from_env(
        workdir=workdir,
        deploy_profile=profile,
        hardware_profile=hardware,
        remote_llm=remote_llm,
        llm_endpoint=llm_endpoint,
        ngc_api_key=ngc_key,
        nvidia_api_key=nvidia_key,
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

@app.command()
def help(
    ctx: typer.Context,
    command: Optional[str] = typer.Argument(None,
        help="Subcommand to drill into (e.g. `spectator help up`). "
             "If omitted, prints a curated end-to-end overview."),
):
    """Curated overview, or `spectator help <command>` for a subcommand's help."""
    if command:
        import click
        click_root = typer.main.get_command(app)
        target = click_root.get_command(ctx, command)
        if target is None:
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("Try [bold]spectator help[/bold] for the list.")
            raise typer.Exit(2)
        # Build a context whose info_name is the target command, so `Usage:`
        # renders as `spectator <command>` rather than `spectator help`.
        sub_ctx = click.Context(target, info_name=f"spectator {command}")
        console.print(target.get_help(sub_ctx))
        return

    console.print("""
[bold cyan]Spectator[/bold cyan] — thin CLI on top of NVIDIA's [link=https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization]Video Search & Summarization Blueprint v3.1[/link].

[bold]Invocation[/bold]: from a checkout of the project, run [italic]./spectator[/italic]
(thin shell wrapper that bootstraps the venv) or [italic]uv run python -m src[/italic]
directly. Spectator is invoked as a Python module, not a console script.

[bold]End-to-end workflow:[/bold]

  [dim]# 1. load credentials (one-time per shell)[/dim]
  [green]export NGC_CLI_API_KEY=...        # https://org.ngc.nvidia.com/setup/api-keys[/green]
  [green]export NVIDIA_API_KEY=...         # https://build.nvidia.com (Get API Key)[/green]

  [dim]# 2. push the tool to the Spark + run user-space install[/dim]
  [green]./spectator deploy    --target <gpu-machine>[/green]

  [dim]# 3. confirm prerequisites[/dim]
  [green]./spectator preflight --target <gpu-machine>[/green]

  [dim]# 4. bring the VSS stack up (in tmux, ~30–45 min first run)[/dim]
  [green]./spectator up        --target <gpu-machine>[/green]
  [green]./spectator logs      --target <gpu-machine> --follow[/green]

  [dim]# 5. process a video (or use the UI)[/dim]
  [green]./spectator ui        --target <gpu-machine>   # ssh -L recipe[/green]
  [green]./spectator process /path/to/video.mp4 --target <gpu-machine>[/green]
  [green]./spectator query "..."                --target <gpu-machine>[/green]

  [dim]# 6. tear down[/dim]
  [green]./spectator down      --target <gpu-machine>[/green]

[bold]Commands[/bold] (run `spectator help <name>` for details):

  [bold]preflight[/bold]   driver / CUDA / docker / nvidia-ctk / NGC checks
  [bold]install[/bold]     clone VSS repo + user-local cache cleaner + NGC docker login
              ([italic]--apply-system[/italic] for nvidia-ctk + usermod + restart docker)
  [bold]rsync[/bold]       rsync this tool to a remote host (just files, no install)
  [bold]deploy[/bold]      full sync: rsync + remote uv sync + remote install
  [bold]up[/bold]          start the VSS stack (in tmux, survives ssh disconnect)
  [bold]down[/bold]        stop the stack
  [bold]status[/bold]      tmux + docker compose ps + UI port + GPU
  [bold]logs[/bold]        tail the bring-up log or a specific docker service
  [bold]ui[/bold]          print the UI URL + ssh-port-forward recipe
  [bold]process[/bold]     upload a [italic]video[/italic] and get a summary back (VSS)
  [bold]query[/bold]       Q&A against indexed videos (OpenAI-compat chat)
  [bold]audio[/bold]       pure-[italic]audio[/italic] transcription via Whisper:
              install / transcribe / status / fetch / presets
  [bold]system[/bold]      cache-cleaner-start / cache-cleaner-stop  (sudo)

[bold]Containment[/bold]: Spectator only writes to its own [italic].venv/[/italic], to [italic]$workdir[/italic] on
the target (default [italic]~/spectator/[/italic]), and to [italic]~/.docker/config.json[/italic] (NGC login).
System mutations are gated behind [italic]--apply-system[/italic].

Friendly walk-through: [italic]README.md[/italic]. Full reference: [italic]REFERENCE.md[/italic].
""".strip())


@app.command()
def preflight(
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="SSH host to check (default: localhost)"),
):
    """Verify VSS prerequisites (driver / CUDA / docker / nvidia-ctk / NGC)."""
    checks = preflight_mod.collect_checks(target)
    ok = preflight_mod.render(checks, target)
    raise typer.Exit(0 if ok else 1)


@app.command()
def install(
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="SSH host to install on (default: local)."),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w",
        help="Where to clone the VSS repo on the target."),
    ngc_key: Optional[str] = typer.Option(None, "--ngc-key",
        envvar="NGC_CLI_API_KEY",
        help="NGC API key (also picked up from $NGC_CLI_API_KEY)."),
    apply_system: bool = typer.Option(False, "--apply-system",
        help="ALSO run system-level mutations (sudo nvidia-ctk runtime configure, "
             "sudo systemctl restart docker, sudo usermod -aG docker). Off by default — "
             "spectator's default install only writes inside $workdir and ~/.docker/config.json."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
):
    """Install the VSS Blueprint (default: user-space only; --apply-system for global mutations)."""
    cfg = _resolve_cfg(workdir, None, None, None, None, ngc_key, None)

    if not skip_preflight:
        console.rule("[bold]preflight[/bold]")
        checks = preflight_mod.collect_checks(target)
        preflight_mod.render(checks, target)
        # don't auto-fail — install can recover some, the user should see the table.

    console.rule(f"[bold]install{' (with --apply-system)' if apply_system else ''}[/bold]")
    if target:
        console.print(f"Installing on [bold]{target}[/bold] ...")
        r = install_mod.install_remote(target, cfg, apply_system=apply_system)
        console.print(r.stdout)
        if not r.ok:
            console.print(f"[red]install failed[/red]\n{r.stderr}")
            raise typer.Exit(1)
    else:
        rc = install_mod.install_local(cfg, apply_system=apply_system)
        if rc != 0:
            raise typer.Exit(rc)


@app.command()
def deploy(
    target: str = typer.Option(..., "--target", "-t", help="SSH host alias."),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
    ngc_key: Optional[str] = typer.Option(None, "--ngc-key", envvar="NGC_CLI_API_KEY"),
    no_install: bool = typer.Option(False, "--no-install",
        help="Just rsync + uv sync, skip the install step."),
):
    """Full sync: rsync source + remote uv sync + remote install."""
    cfg = _resolve_cfg(workdir, None, None, None, None, ngc_key, None)
    r = deploy_mod.deploy(target, cfg, do_install=not no_install)
    raise typer.Exit(0 if r.ok else 1)


@app.command()
def rsync(
    target: str = typer.Option(..., "--target", "-t", help="SSH host."),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """Just rsync the tool to the remote — no uv sync, no install.

    Use during iterative development to push code edits without paying
    the uv-sync round-trip. If you added/changed dependencies, use
    `spectator deploy` instead.
    """
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = deploy_mod.rsync_only(target, cfg)
    raise typer.Exit(0 if r.ok else 1)


@app.command(name="up")
def up_cmd(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
    profile: str = typer.Option(config.DEFAULT_DEPLOY_PROFILE, "--profile", "-p",
        help="dev-profile.sh profile (base / alerts / search / lvs)."),
    hardware: str = typer.Option(config.DEFAULT_HARDWARE_PROFILE, "--hardware", "-H"),
    remote_llm: str = typer.Option(config.DEFAULT_REMOTE_LLM, "--llm",
        help="Remote LLM model name (used with --use-remote-llm)."),
    llm_endpoint: str = typer.Option(config.DEFAULT_LLM_ENDPOINT, "--llm-endpoint",
        envvar="LLM_ENDPOINT_URL"),
    ngc_key: Optional[str] = typer.Option(None, "--ngc-key", envvar="NGC_CLI_API_KEY"),
    nvidia_key: Optional[str] = typer.Option(None, "--nvidia-key", envvar="NVIDIA_API_KEY"),
):
    """Bring the VSS stack up (in tmux, so it survives ssh disconnect)."""
    cfg = _resolve_cfg(workdir, profile, hardware, remote_llm, llm_endpoint, ngc_key, nvidia_key)
    r = stack_mod.up(cfg, host=target)
    console.print(r.stdout)
    if not r.ok:
        console.print(f"[red]up failed[/red]\n{r.stderr}")
    raise typer.Exit(0 if r.ok else 1)


@app.command()
def down(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """Stop the VSS stack."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = stack_mod.down(cfg, host=target)
    console.print(r.stdout)
    raise typer.Exit(0 if r.ok else 1)


@app.command()
def status(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """Show stack health (tmux, docker compose ps, UI port, GPU)."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = stack_mod.status(cfg, host=target)
    console.print(r.stdout)


@app.command()
def logs(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
    service: Optional[str] = typer.Option(None, "--service", "-s",
        help="docker service name to tail (default: spectator's up.log)."),
    follow: bool = typer.Option(False, "--follow", "-f"),
    lines: int = typer.Option(200, "--lines", "-n"),
):
    """Tail the bring-up log (or a specific docker service)."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = stack_mod.logs(cfg, host=target, service=service, follow=follow, lines=lines)
    console.print(r.stdout)


@app.command()
def ui(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
):
    """Print the UI URL (and an `ssh -L` recipe if remote)."""
    ui_port = config.UI_PORT
    api = config.AGENT_API_PORT
    if target:
        console.print(
            f"VSS UI on [bold]{target}[/bold]:\n"
            f"  agent UI:           [bold]http://localhost:{ui_port}[/bold]\n"
            f"  agent REST API:     http://localhost:{api}"
        )
        console.print(
            f"\nTo access from this machine over SSH (forward both):\n\n"
            f"    [green]ssh -N \\\n"
            f"        -L {ui_port}:localhost:{ui_port} \\\n"
            f"        -L {api}:localhost:{api} \\\n"
            f"        {target}[/green]\n\n"
            f"Then open [bold]http://localhost:{ui_port}[/bold] in your browser."
        )
    else:
        console.print(
            f"VSS agent UI:       [bold]http://localhost:{ui_port}[/bold]\n"
            f"VSS Agent REST API: http://localhost:{api}"
        )


@app.command()
def process(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
        help="Path to a video file. If --target is set, must be readable on the remote."),
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="If set, upload to that host's VSS API; otherwise localhost."),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-q",
        help="Optional summarization prompt (default: agent default)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
        help="Write the result JSON here (default: stdout)."),
):
    """Upload a video to the running VSS stack and get back a summary."""
    if target and not video.is_absolute():
        console.print(
            f"[yellow]warning:[/yellow] when --target is set, {video} must be a path "
            f"that VSS can read on {target} (not your local filesystem)."
        )

    if not api.health(target):
        console.print(f"[red]VSS Agent API at port {config.AGENT_API_PORT} is not responding.[/red]")
        console.print(f"  Check `spectator status{' --target ' + target if target else ''}`")
        console.print(f"  If running on remote without a tunnel: `spectator ui --target {target or '<host>'}` for the SSH command.")
        raise typer.Exit(2)

    console.print(f"[bold]→[/bold] uploading {video.name} ...")
    fid = api.upload(video, host=target)
    console.print(f"  file_id = {fid}")

    console.print(f"[bold]→[/bold] summarizing (this can take several minutes for long video) ...")
    result = api.summarize(fid, host=target, prompt=prompt)

    if output:
        import json
        output.write_text(json.dumps(result, indent=2))
        console.print(f"[green]✓[/green] wrote {output}")
    else:
        import json
        console.print_json(data=result)


@app.command()
def query(
    question: str = typer.Argument(..., help="Free-form question."),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    file_ids: Optional[str] = typer.Option(None, "--file-ids",
        help="Comma-separated VSS file ids to scope the query."),
):
    """Ask a Q&A question against the indexed video corpus."""
    if not api.health(target):
        console.print(f"[red]VSS Agent API not responding.[/red]")
        raise typer.Exit(2)
    ids = [s.strip() for s in (file_ids or "").split(",") if s.strip()] or None
    answer = api.query(question, host=target, file_ids=ids)
    console.print(answer)


# ---------------------------------------------------------------------------
# system: cache-cleaner lifecycle (separate from main install path)
# ---------------------------------------------------------------------------

@system_app.command("cache-cleaner-start")
def cache_cleaner_start(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """Start the user-local sys-cache-cleaner.sh in the background (sudo -b)."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = install_mod.start_cache_cleaner(target, cfg)
    console.print(r.stdout)
    if not r.ok:
        console.print(r.stderr)
        raise typer.Exit(1)


@system_app.command("cache-cleaner-stop")
def cache_cleaner_stop(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
):
    """Stop the running sys-cache-cleaner.sh (sudo pkill)."""
    r = install_mod.stop_cache_cleaner(target)
    console.print(r.stdout)


# ---------------------------------------------------------------------------
# audio: pure-audio transcription via Whisper
# ---------------------------------------------------------------------------

@audio_app.command("install")
def audio_install(
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="SSH host to install on (default: local)."),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """Set up the audio-venv (whisper + torch) at $workdir/audio-venv/. Idempotent."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = audio_mod.install_audio_venv(target, cfg)
    console.print(r.stdout)
    if not r.ok:
        console.print(f"[red]install failed[/red]\n{r.stderr}")
        raise typer.Exit(1)


@audio_app.command("presets")
def audio_presets():
    """Show the quality presets and what flags each one applies."""
    audio_mod.render_presets()


@audio_app.command("transcribe")
def audio_transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
        help="Path to an audio file (mp3, wav, m4a, flac, …)."),
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="SSH host to run on (default: local)."),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
    quality: str = typer.Option(audio_mod.DEFAULT_QUALITY, "--quality", "-q",
        help=f"Quality preset: {', '.join(audio_mod.QUALITY_PRESETS)}. "
             f"Run `spectator audio presets` for the table."),
    model: Optional[str] = typer.Option(None, "--model",
        help="Override the preset's whisper model (e.g. large-v3, large-v3-turbo). "
             "For non-English / bilingual audio, prefer large-v3 over large-v3-turbo."),
    language: Optional[str] = typer.Option(None, "--language", "-l",
        help="ISO-639-1 code (en, es, ru, hi, ja, zh, fr, de, pt, …) to lock the "
             "language. Omit (or pass `auto`) for bilingual / code-switched audio "
             "— Whisper auto-detects per 30-second window."),
    task: str = typer.Option("transcribe", "--task",
        help="`transcribe` keeps the original language; `translate` renders the "
             "transcript in English regardless of the source language."),
    clip: Optional[str] = typer.Option(None, "--clip",
        help="Comma-separated start,end pairs in seconds (e.g. \"0,2700\" for first 45 min)."),
    initial_prompt: Optional[str] = typer.Option(None, "--initial-prompt",
        help="Vocabulary-biasing prompt (technical terms, names)."),
    session: Optional[str] = typer.Option(None, "--session",
        help="tmux session name (default: derived from audio filename)."),
    detach: Optional[bool] = typer.Option(None, "--detach/--no-detach",
        help="Run inside tmux on the target. Default: True for --target, False for local."),
    follow: Optional[bool] = typer.Option(None, "--follow/--no-follow",
        help="After starting the tmux session, live-tail the log so progress streams "
             "in your terminal. Ctrl-C exits the tail without stopping the underlying "
             "job. Default: True when running with --target (and --detach), False otherwise."),
    skip_upload: bool = typer.Option(False, "--skip-upload",
        help="Assume the audio file is already at $workdir/audio-in/<basename> on the target."),
    device: Optional[str] = typer.Option(None, "--device",
        help="Force a specific torch device: `cuda` (NVIDIA GPU), `mps` (Apple Silicon), or `cpu`. "
             "Default: auto-detect via the audio-venv's torch (cuda > mps > cpu). Use `--device cpu` "
             "to test the CPU path on a host that has a GPU, or `--device cuda` to force-fail early "
             "if cuda is not actually available."),
):
    """Transcribe an audio file (uploads to target, runs whisper in tmux)."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    if task not in ("transcribe", "translate"):
        console.print(f"[red]--task must be 'transcribe' or 'translate' (got {task!r})[/red]")
        raise typer.Exit(2)
    if device is not None and device not in audio_mod.VALID_DEVICES:
        console.print(f"[red]--device must be one of {audio_mod.VALID_DEVICES} "
                      f"(got {device!r})[/red]")
        raise typer.Exit(2)
    r = audio_mod.transcribe(
        audio, host=target, cfg=cfg,
        quality=quality, model=model, language=language, task=task,
        clip=clip, initial_prompt=initial_prompt,
        session_name=session, detach=detach, follow=follow,
        skip_upload=skip_upload,
        device_override=device,
    )
    # audio_mod.transcribe prints stdout itself (early, before tail -f).
    if not r.ok:
        console.print(f"[red]transcribe failed[/red]\n{r.stderr}")
        raise typer.Exit(1)


@audio_app.command("status")
def audio_status(
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
):
    """List running whisper jobs + completed transcripts."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = audio_mod.status(target, cfg)
    console.print(r.stdout)


@audio_app.command("fetch")
def audio_fetch(
    output: Path = typer.Option(Path("./transcripts"), "--output", "-o",
        help="Local directory to rsync transcripts into."),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    workdir: str = typer.Option(config.DEFAULT_REMOTE_WORKDIR, "--workdir", "-w"),
    only: Optional[str] = typer.Option(None, "--only",
        help="Only fetch one transcript subdirectory (e.g. the audio's stem name)."),
):
    """rsync $workdir/audio-out/ back into a local directory."""
    cfg = _resolve_cfg(workdir, None, None, None, None, None, None)
    r = audio_mod.fetch(target, cfg, output, only=only)
    console.print(r.stdout)
    if not r.ok:
        console.print(f"[red]fetch failed[/red]\n{r.stderr}")
        raise typer.Exit(1)


def main() -> None:
    """Console-script entrypoint for the `spectator` command."""
    app()


if __name__ == "__main__":
    main()
