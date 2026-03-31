# 🚀 Cluster & Environment Guide

This guide covers the essential commands for managing jobs, terminal sessions, and development environments on the cluster.

---

## 🛰️ Slurm Job Management

Use these commands to monitor and manage your cluster workloads.

| Action | Command |
| :--- | :--- |
| **Check Job Status** | `squeue -u <username>` |
| **Cancel Job** | `scancel <jobid>` |

---

## 🪟 Terminal Multiplexing (Tmux)

Tmux allows you to keep your processes running even if your connection drops.

### Session Control
- **New Session**: `tmux new -s <session_name>`
- **Attach**: `tmux attach -t <session_name>`
- **Detach**: `tmux detach` (or `Ctrl + b`, `d`)
- **Kill Session**: `tmux kill-session -t <session_name>`

### Common Shortcuts
> All shortcuts start with the prefix `Ctrl + b`.

| Shortcut | Description |
| :--- | :--- |
| `c` | Create a new window |
| `n` / `p` | Switch to **N**ext / **P**revious window |
| `<number>` | Switch to window by index |
| `%` | Split window **vertically** |
| `"` | Split window **horizontally** |
| `x` | Close current pane/window |

---

## 💻 Remote Development (VS Code)

To set up a remote VS Code session on an allocated node:

1. **SSH** into the main server: `ssh adapt-server`
2. **Allocate** a VS Code session: `luca_vsalloc`
3. **Connect** via VS Code: Select the `luca_vscode` SSH target in your Remote Explorer.

---

## 🧠 Training & GPU Allocation

Heavy computations should be performed on dedicated GPU nodes.

1. **SSH** to the server: `ssh adapt-server`
2. **Allocate GPU**: Run `alloc <gpu_name>` (e.g., `a100`, `rtxa6000`). 
   - *Leave empty for a random available node.*
3. **Launch**: Execute your workload within the allocated shell.

---

## 📦 Package Management (UV)

We use [uv](https://github.com/astral-sh/uv) for extremely fast Python dependency management.

> [!IMPORTANT]
> Always use `uv` to ensure environment consistency.

- **Installation**: `uv sync` (Installs all dependencies from `pyproject.toml`)
- **Execution**: Use `uv run <script.py>` instead of `python <script.py>`.

---
