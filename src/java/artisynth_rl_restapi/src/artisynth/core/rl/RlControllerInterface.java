package artisynth.core.rl;

import java.util.ArrayList;
import java.util.Map;

public interface RlControllerInterface {

	RlState getState();
	double getTime();
	int getActionSize();
	int getObservationSize();
	int getStateSize();
	/** Reset the model and return the initial state. */
	RlState resetState(boolean setExcitationsZero);
	RlState setExcitations(ArrayList<Double> excitations);
	ArrayList<Double> getExcitations();
	ArrayList<Double> getMuscleForces();
	String setSeed(int seed);
	String setTest(boolean isTest);
	boolean getTest();
	/** Return model metadata (sizes, name) in a single call. */
	Map<String, Object> getInfo();
	/** True if the simulation scheduler is currently advancing. */
	boolean isPlaying();
	/** Start the simulation scheduler if it is not already running. */
	void play();
}
