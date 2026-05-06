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

# ── 1b. Check for Maven ───────────────────────────────────────────────────────
MVN="${MVN:-mvn}"
if ! command -v "$MVN" &>/dev/null; then
    echo ""
    echo "ERROR: Maven not found ('$MVN' command missing)."
    echo "  Install it with one of:"
    echo "    sudo apt install maven          # system-wide (Ubuntu/Debian)"
    echo "  or without root:"
    echo "    cd ~"
    echo "    wget https://downloads.apache.org/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz"
    echo "    tar xf apache-maven-3.9.9-bin.tar.gz"
    echo "    export MVN=~/apache-maven-3.9.9/bin/mvn   # add to ~/.bashrc"
    echo "  Then re-run this script."
    exit 1
fi
echo "Using Maven: $("$MVN" --version 2>&1 | head -1)"

# ── 2. Build REST API fat JAR with Maven ──────────────────────────────────────
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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete."
echo ""
echo "IMPORTANT: apply the new CLASSPATH to your current shell:"
echo ""
echo "  source ~/.bashrc"
echo ""
echo "(Or open a new terminal — the entries are already in ~/.bashrc.)"
echo ""
echo "Then launch a training run:"
echo "  artisynth -model artisynth.models.rl.point2point.RlPoint2PointDemo \\"
echo "    '[' -port 8080 -radius 5 ']' -play -noGui"
echo "  cd $SCRIPT_DIR/src/python"
echo "  python main_sb3.py --env Point2PointEnv-v2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
