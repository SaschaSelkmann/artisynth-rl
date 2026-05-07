# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## Language

Communicate with the user in **German**. Write all code, comments, commit messages, and documentation in **English**.

---

## Project overview

ArtiSynth-RL is a reinforcement learning framework built on top of ArtiSynth.
ArtiSynth (JVM) runs as a physics server; a Python Gymnasium client connects
over a local REST API and trains agents with Stable-Baselines3 SAC.

```
artisynth_core/   ← separate repo, must be built and sourced first
artisynth-rl/
├── src/java/
│   ├── artisynth_rl_restapi/   ← Spark REST server (Maven fat JAR)
│   └── artisynth_rl_models/    ← RL model classes (ArtiSynth make)
└── src/python/
    ├── main_sb3.py             ← single-env training / evaluation
    ├── main_sb3_parallel.py    ← parallel training (SubprocVecEnv)
    └── artisynth_envs/         ← Gymnasium env implementations
```

---

## Build commands

```bash
# 1. Source ArtiSynth environment (must cd into artisynth_core first)
cd /path/to/artisynth_core && source setup.bash

# 2. One-shot build and install for artisynth-rl
cd /path/to/artisynth-rl
bash setup.sh          # builds REST API JAR, compiles model classes, installs Python deps
source ~/.bashrc       # apply the new CLASSPATH entries to the current shell

# Build steps individually:
cd src/java/artisynth_rl_restapi && mvn package -DskipTests   # REST API fat JAR
cd src/java/artisynth_rl_models  && make                      # model classes
cd src/python && pip install -e .                              # Python package
```

---

## Running ArtiSynth

ArtiSynth auto-launches when a Python training or evaluation script starts if
nothing is already listening on the target port.  The process is also
automatically terminated when the Python env is closed.

**Manual launch (optional):**
```bash
source setup.bash   # once per shell session

# Point-to-Point
artisynth -model artisynth.models.rl.point2point.RlPoint2PointDemo \
  '[' -port 8080 -radius 5 ']' -play -noGui

# Lumbar Spine
artisynth -model artisynth.models.rl.lumbarspine.RlLumbarSpineDemo \
  '[' -port 8080 ']' -play -noGui

# Jaw
artisynth -model artisynth.models.rl.jaw.RlJawDemo \
  '[' -port 8080 -disc false -condyleConstraints false -condylarCapsule true ']' \
  -play -noGui
```

Drop `-noGui` to show the 3-D viewer, or pass `--gui` to the Python script.

---

## Training and evaluation

```bash
cd src/python

# Single env
python main_sb3.py --env Point2PointEnv-v2 --timesteps 500000

# Parallel (4 workers, ports 8080-8083)
python main_sb3_parallel.py --env Point2PointEnv-v2 --n_envs 4 --timesteps 500000

# Evaluate
python main_sb3.py --env Point2PointEnv-v2 \
  --load results/Point2PointEnv-v2/sac_parallel --test --test_episodes 20

# GUI (restarts headless server if already running)
python main_sb3.py --env Point2PointEnv-v2 --gui --test \
  --load results/Point2PointEnv-v2/sac_parallel
```

Boolean flags use `--flag` / `--no-flag` syntax (e.g. `--no-gui`, `--no-test`).

# TensorBoard (enabled by default)
tensorboard --logdir tb_logs   # open http://localhost:6006
# Pass --tb_log '' to disable logging.

---

## Architecture

### Java side (`artisynth_rl_restapi` + `artisynth_rl_models`)

| Class | Role |
|---|---|
| `RlRestApi` | Spark HTTP server; exposes REST endpoints |
| `RlControllerInterface` | Interface between REST API and model |
| `RlController` | ControllerBase; applies excitations, thread-safe reset via `resetPending` flag |
| `RlModelInterface` | Implement this in your RootModel |
| `RlPoint2PointDemo` | Simple 1-D / 2-D / 3-D tracking model |
| `RlLumbarSpineDemo` | 6-vertebra lumbar spine model |
| `RlJawDemo` | FEM jaw model |

**Thread safety**: `RlController.resetState()` is called from the HTTP (Spark)
thread. The actual `setState()` / `initialize()` / `randomizeTarget()` calls
happen inside `apply()` on the simulation thread via a `resetPending` volatile
flag + `Object.wait()` / `notifyAll()` handshake.

### Python side (`src/python`)

| Class | Role |
|---|---|
| `ArtiSynthBase` | `gym.Env` base; REST client, auto-launch, `close()` terminates ArtiSynth |
| `Point2PointEnv` | Simple tracking env; reward based on distance improvement |
| `SpineEnv` | Lumbar spine env; weighted distance + effort + smoothness reward |
| `JawEnv` | Jaw env; log-distance reward with optional symmetry penalty |

All envs follow the Gymnasium v0.26+ API: `step()` returns
`(obs, reward, terminated, truncated, info)`, `reset()` accepts `seed` and `options`.

### REST API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/info` | Action / obs / state sizes and model name |
| `GET` | `/state` | Full environment state |
| `GET` | `/isPlaying` | Whether the scheduler is running |
| `POST` | `/excitations` | Apply excitations, return new state |
| `POST` | `/reset` | Thread-safe episode reset |
| `POST` | `/play` | Start the scheduler if paused |
| `POST` | `/setSeed` | Set Java RNG seed |
| `POST` | `/setTest` | Toggle test mode |

---

## Key constraints

- `setState()` and `initialize()` **must** be called from the ArtiSynth
  simulation thread, never from an HTTP handler.
- `source setup.bash` uses `pwd` — always `cd` into `artisynth_core` first.
- `setup.bash` sets `ARTISYNTH_HOME`, `CLASSPATH`, and `PATH`; `~/.bashrc`
  adds the RL-specific JARs on top. Both must be sourced.
- ArtiSynth startup logs go to `artisynth_<port>.log` in the working directory.

---

## Workflow

- Prepare commits (`git add` + `git commit`); the user handles `git push`.
- Keep commit messages in English, imperative, ≤ 72 chars per line.
- Do not skip pre-commit hooks (`--no-verify`).
- Do not push to `main` or force-push without explicit user instruction.
