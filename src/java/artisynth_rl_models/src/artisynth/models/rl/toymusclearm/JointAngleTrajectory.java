package artisynth.models.rl.toymusclearm;

import java.util.Random;

/**
 * Reference joint-angle trajectory used by {@link RlToyMuscleArmDemo} as the
 * tracking target. Implementations are pure functions of simulation time and
 * therefore deterministic between calls to {@link #randomize(Random)}.
 *
 * <p>All angles and angular velocities are in SI units (rad, rad/s).
 */
public interface JointAngleTrajectory {

   /**
    * Returns the reference joint angles {@code [theta0, theta1]} (rad) at
    * simulation time {@code t} (s).
    */
   double[] angles(double t);

   /**
    * Returns the reference joint angular velocities
    * {@code [thetaDot0, thetaDot1]} (rad/s) at simulation time {@code t} (s).
    */
   double[] velocities(double t);

   /**
    * Resamples internal parameters (e.g. amplitudes, frequencies, phases) from
    * {@code rng}. Called by the target controller at the start of each
    * episode.
    */
   void randomize(Random rng);
}
