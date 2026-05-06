package artisynth.core.rl;

public interface RlTargetControllerInterface {
	void reset();

	/**
	 * Directly apply episode-specific randomization (new target position, angle,
	 * etc.) and clear any dependent dynamic quantities (velocity, cached state).
	 * Called immediately after the model snapshot has been restored, so
	 * implementations must not assume any particular simulation time.
	 */
	void randomizeTarget();
}
