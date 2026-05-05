#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Validate ARTISYNTH_HOME ────────────────────────────────────────────────
if [ -z "$ARTISYNTH_HOME" ]; then
    echo "ERROR: ARTISYNTH_HOME is not set."
    echo "  cd to artisynth_core and run:  source setup.bash"
    exit 1
fi
echo "Using ARTISYNTH_HOME=$ARTISYNTH_HOME"

# ── 2. Build REST API fat JAR with Maven 3 ────────────────────────────────────
MVN="${MVN:-mvn}"
echo "Building artisynth_rl_restapi …"
cd "$SCRIPT_DIR/src/java/artisynth_rl_restapi"
$MVN package -q
RESTAPI_JAR="$SCRIPT_DIR/src/java/artisynth_rl_restapi/target/artisynth_rl_restapi-2.0.0.jar"
echo "  JAR: $RESTAPI_JAR"

# ── 3. Compile artisynth_rl_models ────────────────────────────────────────────
echo "Compiling artisynth_rl_models …"
cd "$SCRIPT_DIR/src/java/artisynth_rl_models"
make
MODELS_CLASSES="$SCRIPT_DIR/src/java/artisynth_rl_models/classes"

# ── 4. Update CLASSPATH ───────────────────────────────────────────────────────
add_to_classpath() {
    local entry="$1"
    if [[ ":$CLASSPATH:" != *":$entry:"* ]]; then
        export CLASSPATH="$entry:$CLASSPATH"
        echo "  Added to CLASSPATH: $entry"
    fi
}

add_to_classpath "$MODELS_CLASSES"
add_to_classpath "$RESTAPI_JAR"

# Persist to ~/.bashrc if not already there
for entry in "$MODELS_CLASSES" "$RESTAPI_JAR"; do
    line="export CLASSPATH=\"$entry:\$CLASSPATH\""
    grep -qF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
done

# ── 5. Python dependencies ────────────────────────────────────────────────────
echo "Installing Python dependencies …"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Setup complete. To launch a training run:"
echo "  cd $SCRIPT_DIR/src/python"
echo "  python main_sb3.py --env Point2PointEnv-v2"
