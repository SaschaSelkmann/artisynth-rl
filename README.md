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

## Prerequisites

| Dependency | Minimum | Notes |
|---|---|---|
| Java (JDK) | 17 | Must be on `PATH` as `javac`/`java` |
| Maven | 3.9 | See below if not installed system-wide |
| Python | 3.11 | Conda or system |
| ArtiSynth | latest | Must be built; `$ARTISYNTH_HOME` must point to `artisynth_core/` |

### Install Maven 3.9 without root

```bash
cd ~
wget https://downloads.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz
tar xf apache-maven-3.9.6-bin.tar.gz
export MVN=~/apache-maven-3.9.6/bin/mvn   # add to ~/.bashrc
```

---

## Installation

### 1. Build and source ArtiSynth

```bash
cd /path/to/artisynth_core
make
source setup.bash   # sets ARTISYNTH_HOME, CLASSPATH, PATH
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

### 3. Install the Python package

```bash
cd src/python
pip install -e .
```

---

## Running a training session

### Step 1 — Launch ArtiSynth

Open a terminal and start the simulation server. Model-specific args go inside `'[' ... ']'` (single-quoted brackets so the shell does not interpret them):

**Point-to-Point (simplest model):**
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

Drop `-noGui` to show the 3-D viewer.

### Step 2 — Train

```bash
cd src/python
python main_sb3.py --env Point2PointEnv-v2 --timesteps 500000
```

Checkpoints are saved every 10 000 steps to `results/Point2PointEnv-v2/`. The final model is saved to `results/Point2PointEnv-v2/sac`.

### Step 3 — Evaluate

```bash
python main_sb3.py --env Point2PointEnv-v2 \
  --load results/Point2PointEnv-v2/sac \
  --test --test_episodes 20
```

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

**`ARTISYNTH_HOME` not set / model class not found**
`setup.bash` relies on `pwd`. Always `cd /path/to/artisynth_core && source setup.bash` before building or running.

**`ClassNotFoundException` when launching a model**
The fat JAR and model classes must both be on `$CLASSPATH`. Re-run `bash setup.sh` or export them manually:
```bash
export CLASSPATH=/path/to/artisynth_rl_restapi-2.0.0.jar:/path/to/artisynth_rl_models/classes:$CLASSPATH
```

**Simulation time freezes / training hangs**
A tight HTTP polling loop can starve the JVM scheduler and freeze simulation time. Keep `--wait_action 0.0` (the default). If you call `_sleep_sim` in a custom environment, ensure there is a `time.sleep(0.001)` yield inside the loop.

**Port already in use**
Kill the previous ArtiSynth process or pick a different port and pass the same value to both sides:
```bash
artisynth -model … '[' -port 8081 ']' -play -noGui &
python main_sb3.py --env Point2PointEnv-v2 --port 8081
```
