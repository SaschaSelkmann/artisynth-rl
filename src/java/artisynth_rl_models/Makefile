RL_ROOT     := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))../../..)
RESTAPI_JAR := $(RL_ROOT)/src/java/artisynth_rl_restapi/target/artisynth_rl_restapi-2.0.0.jar

ifndef ARTISYNTH_HOME
  $(error ARTISYNTH_HOME is not set. cd to artisynth_core and run 'source setup.bash' first.)
endif

SRCDIR  := src
OUTDIR  := classes
SOURCES := $(shell find $(SRCDIR) -name "*.java")
CP      := $(ARTISYNTH_HOME)/classes:$(ARTISYNTH_HOME)/lib/*:$(RESTAPI_JAR)

.PHONY: all clean

all: $(OUTDIR)
	javac --release 11 -encoding UTF-8 \
	      -classpath "$(CP)" \
	      -sourcepath $(SRCDIR) \
	      -d $(OUTDIR) \
	      $(SOURCES)
	@echo "artisynth_rl_models compiled to $(OUTDIR)/"

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)
