# ArtiSynth-RL

Reinforcement learning framework for [ArtiSynth](https://www.artisynth.org) biomechanical simulations. ArtiSynth runs as a physics server; a Python Gymnasium client connects over a local REST API to train agents with Stable-Baselines3.

## Architecture

```
┌─────────────────────────────────┐      HTTP / JSON      ┌─────────────────────────┐
│  ArtiSynth (JVM)                │◄─────────────────────►│  Python (Gymnasium)     │
│  RlModelInterface               │  :8080                │  ArtiSynthBase (gym.Env)│
│  RlController                   │                       │  Point2PointEnv         │
│  Spark REST API v2              │                       │  Stable-Baselines3 SAC  │
└─────────────────────────────────┘                       └─────────────────────────┘
```

The Java side implements the physics and exposes muscle excitations + state via REST. The Python side wraps this as a standard Gymnasium environment.

---

## How the framework works

### Action space and exploration

The action space is the vector of muscle excitations, bounded to `[0, 1]` per muscle (or `[-0.1, +0.1]` for incremental actions). These bounds are defined in `get_state_boundaries()` inside each env class and passed to Gymnasium's `spaces.Box`.

SAC explores by learning a Gaussian policy over this space. With `--ent_coef auto` the entropy coefficient is tuned automatically — no manual exploration schedule is needed.

```python
# From ArtiSynthBase.init_spaces()
low_a  = c.LOW_EXCITATION   # 0.0
high_a = c.HIGH_EXCITATION  # 1.0
self.action_space = spaces.Box(low=low_a, high=high_a, shape=(action_size,), dtype=np.float32)
```

### Reward function

The reward function lives entirely in the Python env class. SB3 only ever sees the scalar return value — the full ArtiSynth state dict (positions, velocities, muscle forces, time) is available inside the env to compute any reward signal you need.

```python
# Point2PointEnv: reward progress toward the target
def _calculate_reward(self, new_dist, prev_dist):
    if new_dist < self.goal_threshold:
        return self.goal_reward, True, {}          # goal reached
    if prev_dist - new_dist > 0:
        return 1.0 / self.episode_counter, False, {}  # moving closer
    return -1.0, False, {}                         # moving away

# SpineEnv: weighted sum of distance, effort, and smoothness
reward = -(phi_u * w_u + phi_d * w_d + phi_r * w_r) / 10

# JawEnv: log-distance with optional bilateral symmetry penalty
reward = -w_u * log10(distance + ε) - w_r * muscle_forces - w_s * symmetry_loss
```

Weights (`--w_u`, `--w_d`, `--w_r`, `--w_s`) are passed as command-line arguments.

### Algorithm choice

SAC is the default because it handles continuous action spaces (muscle excitations) well and requires no manual exploration schedule. It is not hardcoded into Gymnasium — swapping it for another SB3 algorithm requires changing one line:

```python
# main_sb3.py — replace SAC with TD3 or PPO
from stable_baselines3 import TD3
model = TD3('MlpPolicy', env, ...)
```

Any algorithm that accepts a continuous-action Gymnasium env will work.

### Combining with other control principles

Gymnasium is a thin wrapper. Classical controllers can be integrated at several levels:

**Inside the env** — run a secondary controller in `step()` and fold its output into the reward or observation. The RL agent sees only the net effect:

```python
def step(self, action):
    pid_correction = self.pid.compute(self.get_state_dict())
    blended = 0.7 * action + 0.3 * pid_correction   # blend RL + PID
    return super().step(blended)
```

**Inside ArtiSynth** — register additional `ControllerBase` instances alongside `RlController`. They run in the same simulation step and can, for example, enforce joint limits or apply passive stabilising forces before the RL excitations are applied.

```java
// In addRlController()
addController(new PassiveStabiliser(mech));  // runs before RlController
addController(rlController);
```

**Hierarchical control** — a high-level RL agent sets sub-goals; a low-level classical controller (inverse kinematics, PD) executes them. The high-level env calls `step()` on the low-level env and treats its convergence as a single action.

### Curriculum learning

Increase task difficulty over training by varying the target in `resetState()` on the Java side or by modifying `reset()` in the Python env:

```python
def reset(self, *, seed=None, options=None):
    # Gradually expand the target radius as training progresses
    progress = self.num_resets / self.total_resets
    self.current_radius = self.min_radius + progress * (self.max_radius - self.min_radius)
    self.net.send_msg({'radius': self.current_radius}, ...)
    return super().reset(seed=seed)
```

---

## Prerequisites

| Dependency | Minimum | Notes |
|---|---|---|
| Java (JDK) | 17 | Must be on `PATH` as `javac`/`java` |
| Maven | 3.8+ | `sudo apt install maven` or see below for a root-free install |
| Python | 3.11 | Conda or system |
| ArtiSynth | latest | Must be built; `$ARTISYNTH_HOME` must point to `artisynth_core/` |

### Install Maven

**With root (recommended):**
```bash
sudo apt install maven   # Ubuntu/Debian — installs Maven 3.8+
```

**Without root:**
```bash
cd ~
wget https://downloads.apache.org/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz
tar xf apache-maven-3.9.9-bin.tar.gz
export MVN=~/apache-maven-3.9.9/bin/mvn   # add to ~/.bashrc
```

---

## Installation

### 1. Build and source ArtiSynth

```bash
cd /path/to/artisynth_core
source setup.bash   # sets ARTISYNTH_HOME, CLASSPATH, PATH
make
```

`setup.bash` uses `pwd`, so **you must `cd` into `artisynth_core` first**.

### 2. Run the setup script

```bash
cd /path/to/artisynth-rl
source /path/to/artisynth_core/setup.bash   # if not already sourced
bash setup.sh
```

`setup.sh` will:
1. Build the REST API fat JAR (`artisynth_rl_restapi-2.0.0.jar`) with Maven 3
2. Compile all RL model classes with `make`
3. Add both to `$CLASSPATH` (and persist the entries to `~/.bashrc`)
4. Install Python dependencies from `requirements.txt`

### 3. Apply the new CLASSPATH to your shell

`setup.sh` adds two `export CLASSPATH=…` lines to `~/.bashrc`. They are **not** active in the terminal where you just ran `setup.sh` (because `bash setup.sh` runs in a subprocess). Apply them now:

```bash
source ~/.bashrc
```

Or simply open a new terminal — the entries are already persisted.

### 4. Install the Python package

```bash
cd src/python
pip install -e .
```

---

## Running a training session

### Available environments

| Gymnasium ID | ArtiSynth model | Observation | RAM / instance |
|---|---|---|---|
| `Point2PointEnv-v2` | `RlPoint2PointDemo` | point position (+ excitations) | ~200 MB |
| `Point2PointEnv-v3` | `RlPoint2PointDemo` (3-D) | point position (+ excitations) | ~200 MB |
| `SpineEnv-v0` | `RlLumbarSpineDemo` | 6 vertebra orientations (+ excitations) | ~500 MB |
| `JawEnv-v0` | `RlJawDemo` | lower incisor position (+ excitations) | ~1.5 GB |
| `JawEnv-v1` | `RlJawDemo` (capsule) | lower incisor position (+ excitations) | ~1.5 GB |
| `JawEnv-v2` | `RlJawDemo` (capsule + velocity) | position + velocity (+ excitations) | ~1.5 GB |

> **Parallelisation note**: RAM scales linearly with `--n_envs`. For Jaw models limit to 2–3 workers; for Point2Point up to 8 workers is practical.

### Step 1 — Launch ArtiSynth (optional)

ArtiSynth auto-launches when the Python script starts if nothing is already listening on the target port. Manual launch is only needed if you want to control the startup flags yourself:

**Point-to-Point:**
```bash
artisynth -model artisynth.models.rl.point2point.RlPoint2PointDemo \
  '[' -port 8080 -radius 5 ']' -play -noGui
```

**Lumbar Spine:**
```bash
artisynth -model artisynth.models.rl.lumbarspine.RlLumbarSpineDemo \
  '[' -port 8080 ']' -play -noGui
```

**Jaw:**
```bash
artisynth -model artisynth.models.rl.jaw.RlJawDemo \
  '[' -port 8080 -disc false -condyleConstraints false -condylarCapsule true ']' \
  -play -noGui
```

Drop `-noGui` to show the 3-D viewer. Pass `--gui` to the Python script to let it auto-launch ArtiSynth with a viewer.

### Step 2 — Train

**Single environment:**
```bash
cd src/python
python main_sb3.py --env Point2PointEnv-v2 --timesteps 500000
python main_sb3.py --env SpineEnv-v0       --timesteps 1000000
python main_sb3.py --env JawEnv-v1         --timesteps 2000000
```

Checkpoints are saved every 10 000 steps to `results/<env>/`. The final model is saved to `results/<env>/sac`.

**Parallel environments (faster training):**
```bash
python main_sb3_parallel.py --env Point2PointEnv-v2 --n_envs 4 --timesteps 500000
python main_sb3_parallel.py --env SpineEnv-v0       --n_envs 2 --timesteps 1000000
python main_sb3_parallel.py --env JawEnv-v1         --n_envs 2 --timesteps 2000000
```

Each worker auto-launches its own ArtiSynth instance on `--port + rank`. The final model is saved to `results/<env>/sac_parallel`. ArtiSynth processes are terminated automatically when training ends.

### Step 3 — Evaluate

```bash
python main_sb3.py --env Point2PointEnv-v2 \
  --load results/Point2PointEnv-v2/sac_parallel \
  --test --test_episodes 20
```

Add `--gui` to open the ArtiSynth viewer during evaluation (restarts the server if it was running headless).

### Demo videos

[![Point to point tracking](https://img.youtube.com/vi/UqHt4KbsaII/0.jpg)](https://www.youtube.com/watch?v=UqHt4KbsaII)
[![Jaw model demo](https://img.youtube.com/vi/E9Ix0q5frSQ/0.jpg)](https://www.youtube.com/watch?v=E9Ix0q5frSQ)

---

## `main_sb3.py` reference

### Environment options

| Flag | Default | Description |
|---|---|---|
| `--env` | `Point2PointEnv-v2` | Gymnasium env ID |
| `--ip` | `localhost` | ArtiSynth host |
| `--port` | `8080` | REST API port (must match `-port` in the launch command) |
| `--gui` | off | Show ArtiSynth viewer when auto-launching |
| `--seed` | `12345` | RNG seed |
| `--include_current_state` | on | Append current position to observation |
| `--include_current_excitations` | on | Append muscle excitations to observation |
| `--incremental_actions` | off | Actions are deltas added to current excitations |
| `--zero_excitations_on_reset` | on | Zero all muscles on episode reset |
| `--goal_threshold` | `0.1` | Distance (m) at which the episode is solved |
| `--goal_reward` | `5.0` | Reward given on goal reached |
| `--reset_step` | `200` | Max steps per episode before truncation |
| `--wait_action` | `0.0` | Simulation seconds to advance after each action |

### SAC hyper-parameters

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | `500000` | Total environment steps |
| `--lr` | `3e-4` | Learning rate |
| `--batch_size` | `256` | Mini-batch size |
| `--buffer_size` | `100000` | Replay buffer capacity |
| `--learning_starts` | `1000` | Steps before the first gradient update |
| `--tau` | `0.005` | Soft target update coefficient |
| `--gamma` | `0.99` | Discount factor |
| `--ent_coef` | `auto` | Entropy coefficient (`auto` or a float) |

### I/O

| Flag | Default | Description |
|---|---|---|
| `--save_path` | `results/<env>/sac` | Path prefix for the saved model |
| `--load` | — | Path to a saved model; resumes training or runs evaluation |
| `--test` | off | Evaluation mode (requires `--load`) |
| `--test_episodes` | `10` | Number of episodes to run in evaluation mode |

---

## REST API v2 reference

The Spark HTTP server starts on the configured port when ArtiSynth launches the RL model. All responses are JSON.

### Endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/` | — | Health check, returns API version string |
| `GET` | `/info` | — | Model metadata: action/obs/state sizes and name |
| `GET` | `/state` | — | Full environment state (see below) |
| `GET` | `/time` | — | Current simulation time in seconds |
| `GET` | `/obsSize` | — | Observation vector length |
| `GET` | `/stateSize` | — | State vector length |
| `GET` | `/actionSize` | — | Number of controllable muscles |
| `GET` | `/excitations` | — | Current muscle excitation values |
| `POST` | `/excitations` | `{"excitations":[…]}` | Apply excitations and return new state |
| `POST` | `/reset` | `true` or `false` | Reset episode; body controls whether excitations are zeroed |
| `POST` | `/setSeed` | `12345` | Set RNG seed |
| `POST` | `/setTest` | `true` or `false` | Toggle test mode (disables randomization) |

### `/info` response

```json
{
  "actionSize": 8,
  "obsSize": 12,
  "stateSize": 28,
  "name": "InvTracker"
}
```

### `/state` response

```json
{
  "observation": {
    "point":     { "position": [x, y, z], "velocity": [vx, vy, vz] },
    "point_ref": { "position": [x, y, z], "velocity": [vx, vy, vz] }
  },
  "properties": {},
  "excitations":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "muscleForces": [f0, f1, "…"],
  "time": 1.24,
  "terminated": false,
  "truncated": false
}
```

`terminated` and `truncated` are set by `RlState` on the Java side and can be read by custom Python environments to avoid an extra HTTP round-trip.

---

## Building a custom RL model

### 1. Implement `RlModelInterface` in your `RootModel`

```java
public class MyRlModel extends RootModel implements RlModelInterface {

    private int port = 8080;
    private RlController rlController;
    private MyTargetController targetController;

    @Override
    public void build(String[] args) throws IOException {
        parseArgs(args);
        // build your MechModel, add muscles, geometry, etc.
        addRlController();
    }

    @Override
    public void parseArgs(String[] args) {
        // parse -port and any model-specific flags
        for (int i = 0; i < args.length; i++) {
            if ("-port".equals(args[i])) port = Integer.parseInt(args[++i]);
        }
    }

    @Override
    public void addRlController() {
        rlController = new RlController("InvTracker", this);
        // register controllable muscles:
        for (MuscleExciter e : myExciters) {
            rlController.addExciter(e);
        }
        // register current and target tracked components:
        rlController.addMotionTarget(currentPoint);
        rlController.addMotionTarget(targetPoint);
        addController(rlController);
        // start the REST server:
        new RlRestApi(rlController, port);
    }

    @Override
    public ArrayList<RlProp> getRlProps() {
        ArrayList<RlProp> props = new ArrayList<>();
        props.add(new RlProp("point",     currentPoint, "position", "velocity"));
        props.add(new RlProp("point_ref", targetPoint,  "position", "velocity"));
        return props;
    }

    @Override
    public RlTargetControllerInterface getTargetMotionController() {
        return targetController;
    }

    @Override
    public void resetState() {
        // randomize target position, reset rigid-body poses, etc.
    }

    @Override
    public double getTime() {
        return myMechModel.getTime();
    }
}
```

### 2. Register a Gymnasium environment

Add an entry in `src/python/artisynth_envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='MyEnv-v1',
    entry_point='artisynth_envs.envs.my_env:MyEnv',
    kwargs={
        'artisynth_model': 'artisynth.models.rl.mymodel.MyRlModel',
        'artisynth_args': '-radius 5',
        'goal_threshold': 0.05,
        'goal_reward': 10.0,
        'reset_step': 300,
        'wait_action': 0.0,
        # ... other ArtiSynthBase defaults ...
    },
)
```

Then subclass `Point2PointEnv` (or `ArtiSynthBase` directly) and override `_calculate_reward` and `get_state_boundaries` as needed.

### 3. Launch and train

```bash
artisynth -model artisynth.models.rl.mymodel.MyRlModel '[' -port 8080 ']' -play -noGui
python main_sb3.py --env MyEnv-v1
```

---

## Project layout

```
artisynth-rl/
├── setup.sh                          # one-shot build + install
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment spec
└── src/
    ├── java/
    │   ├── artisynth_rl_restapi/     # Spark REST server (built as fat JAR)
    │   │   └── src/artisynth/core/rl/
    │   │       ├── RlRestApi.java            # HTTP routes
    │   │       ├── RlControllerInterface.java
    │   │       ├── RlState.java              # response DTO
    │   │       └── RlStateSerializer.java    # Gson adapter
    │   └── artisynth_rl_models/      # RL-aware ArtiSynth models
    │       └── src/artisynth/
    │           ├── core/rl/
    │           │   ├── RlController.java     # implements RlControllerInterface
    │           │   ├── RlModelInterface.java # implement this in your RootModel
    │           │   └── RlProp.java           # property descriptor for observations
    │           └── models/rl/
    │               ├── point2point/RlPoint2PointDemo.java
    │               ├── lumbarspine/RlLumbarSpineDemo.java
    │               └── jaw/RlJawDemo.java
    └── python/
        ├── main_sb3.py               # training / evaluation entry point
        ├── artisynth_envs/
        │   ├── __init__.py           # Gymnasium env registration
        │   ├── artisynth_base_env.py # base Gymnasium class (REST client)
        │   └── envs/
        │       └── point2point_env.py
        └── common/
            ├── rest_client.py
            └── constants.py
```

---

## Troubleshooting

**`mvn: command not found` during setup**
Maven is not installed. Install it with `sudo apt install maven` (Ubuntu/Debian) or follow the root-free install steps in the Prerequisites section above, then re-run `bash setup.sh`.

**`ARTISYNTH_HOME` not set / model class not found**
`setup.bash` relies on `pwd`. Always `cd /path/to/artisynth_core && source setup.bash` before building or running.

**`ClassNotFoundException` when launching a model**
The fat JAR and model classes must be on `$CLASSPATH`. After running `bash setup.sh`, you must also apply the changes to your current shell:
```bash
source ~/.bashrc
```
Or open a new terminal. Re-run `bash setup.sh` first if you skipped it, then source. You can also export manually:
```bash
export CLASSPATH=/path/to/artisynth_rl_restapi-2.0.0.jar:/path/to/artisynth_rl_models/classes:$CLASSPATH
```

**Simulation time freezes / training hangs**
A tight HTTP polling loop can starve the JVM scheduler and freeze simulation time. Keep `--wait_action 0.0` (the default). If you call `_sleep_sim` in a custom environment, ensure there is a `time.sleep(0.001)` yield inside the loop.

**Port already in use**
A previous ArtiSynth run didn't exit cleanly. Free the port before relaunching:
```bash
kill $(lsof -ti:8080) 2>/dev/null; true
```
When Python auto-launches ArtiSynth (`run_artisynth`), this is handled automatically. For manual launches you must run the command above yourself, or pick a different port and pass the same value to both sides:
```bash
artisynth -model … '[' -port 8081 ']' -play -noGui &
python main_sb3.py --env Point2PointEnv-v2 --port 8081
```

**Training freezes / model stops responding**
ArtiSynth is a separate JVM process; if it hangs, the Python side used to wait forever on the HTTP call. Now every request has a 30-second timeout — you will see a `requests.exceptions.Timeout` error instead of a silent freeze. If this happens regularly, check the ArtiSynth console for JVM errors, or reduce simulation complexity.
