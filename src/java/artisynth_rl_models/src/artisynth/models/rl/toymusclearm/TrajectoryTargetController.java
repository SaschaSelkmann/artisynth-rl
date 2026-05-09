package artisynth.models.rl.toymusclearm;

import java.util.Random;

import artisynth.core.modelbase.ControllerBase;
import artisynth.core.rl.RlTargetControllerInterface;

/**
 * Controller that drives the per-step reference joint angles for the
 * {@link RlToyMuscleArmDemo} from a {@link JointAngleTrajectory}.
 *
 * <p>The controller is purely virtual: it does not modify any
 * {@code MechModel} state. Instead, it caches the latest reference angles
 * and angular velocities every simulation step so that the model's
 * {@code getRlProps()} can publish them as part of the RL observation.
 */
public class TrajectoryTargetController extends ControllerBase
   implements RlTargetControllerInterface {

   private final JointAngleTrajectory trajectory;
   private final Random rng;

   private double[] currentAngles = new double[] { 0.0, 0.0 };       // rad
   private double[] currentVelocities = new double[] { 0.0, 0.0 };   // rad/s

   private volatile boolean resetPending = false;

   public TrajectoryTargetController (
      JointAngleTrajectory trajectory, Random rng) {
      this.trajectory = trajectory;
      this.rng = rng;
      // Prime cache with t=0 values so the first /state read is consistent.
      this.currentAngles = trajectory.angles (0.0);
      this.currentVelocities = trajectory.velocities (0.0);
   }

   public JointAngleTrajectory getTrajectory() {
      return trajectory;
   }

   /** Latest reference joint angles {@code [theta0, theta1]} (rad). */
   public double[] getCurrentAngles() {
      return currentAngles.clone();
   }

   /** Latest reference angular velocities {@code [thetaDot0, thetaDot1]} (rad/s). */
   public double[] getCurrentVelocities() {
      return currentVelocities.clone();
   }

   @Override
   public void reset() {
      resetPending = true;
   }

   @Override
   public void randomizeTarget() {
      trajectory.randomize (rng);
      // Re-prime cache at t=0 since resetState() restores simulation time to 0.
      currentAngles = trajectory.angles (0.0);
      currentVelocities = trajectory.velocities (0.0);
   }

   @Override
   public void apply (double t0, double t1) {
      if (resetPending) {
         resetPending = false;
         randomizeTarget();
         return;
      }
      currentAngles = trajectory.angles (t1);
      currentVelocities = trajectory.velocities (t1);
   }
}
