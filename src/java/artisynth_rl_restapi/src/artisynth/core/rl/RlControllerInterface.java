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

	/**
	 * Begin grabbing viewer frames. Requires GUI mode (a viewer must exist).
	 * @param name      base name of the rendered movie file (no extension).
	 * @param outputDir absolute path to the output folder (created if missing).
	 *                  May be {@code null} to keep MovieMaker's default.
	 * @param fps       target frame rate in frames per second.
	 * @return short status message.
	 */
	String startRecording(String name, String outputDir, double fps);

	/**
	 * Stop grabbing frames and assemble them into a movie file using
	 * MovieMaker's configured method (FFMPEG by default). Returns
	 * {@code frames}, {@code rendered}, {@code outputDir}, {@code file}.
	 */
	Map<String, Object> stopRecording();
}
