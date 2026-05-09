package artisynth.core.rl;

import java.util.ArrayList;

import artisynth.core.driver.Main;
import artisynth.core.driver.Scheduler;

/**
 * Synchronous, deterministic stepping: each submitted excitation vector
 * triggers exactly {@code subStepsPerAction} simulator iterations, after
 * which the scheduler is paused and the resulting state is returned.
 *
 * <p>Per RL action: stage excitations → {@link Scheduler#playRequest}
 * to run for {@code subStepsPerAction × stepSize} seconds → spin until
 * {@link Scheduler#isPlaying()} returns false → read state.
 *
 * <p>The strategy assumes a single (HTTP) caller thread per controller and
 * does not serialise concurrent submissions. Configuration (sub-step count,
 * action duration) is fixed at construction time; create a new instance to
 * change either.
 */
public class SteppedStrategy implements StepStrategy {

   public static final String NAME = "stepped";

   private final int subStepsPerAction;

   public SteppedStrategy (int subStepsPerAction) {
      if (subStepsPerAction < 1) {
         throw new IllegalArgumentException (
            "subStepsPerAction must be >= 1, got " + subStepsPerAction);
      }
      this.subStepsPerAction = subStepsPerAction;
   }

   /**
    * Convenience constructor: derive the sub-step count from a desired
    * action duration in seconds and the simulator step size in seconds
    * (rounded up so the actual duration is never less than requested).
    */
   public static SteppedStrategy fromDuration (
      double waitActionSec, double stepSizeSec) {
      int n = (int) Math.max (1, Math.ceil (waitActionSec / stepSizeSec));
      return new SteppedStrategy (n);
   }

   public int getSubStepsPerAction() {
      return subStepsPerAction;
   }

   @Override
   public String name() {
      return NAME;
   }

   @Override
   public boolean isSynchronous() {
      return true;
   }

   @Override
   public void initialize (RlController controller) {
      Scheduler s = getScheduler();
      if (s != null) {
         s.pause(); // expected baseline state for stepped mode
      } else {
         Log.info ("SteppedStrategy.initialize: scheduler not yet available; "
                   + "first applyExcitations() will pause it as needed.");
      }
   }

   @Override
   public RlState applyExcitations (
      RlController controller, ArrayList<Double> excitations) {
      Scheduler s = getScheduler();
      if (s == null) {
         Log.info ("SteppedStrategy: no scheduler — falling back to FreeRun "
                   + "for this action.");
         controller.stageExcitations (excitations);
         return new RlState();
      }

      // Stage excitations first; the next apply() inside the playRequest
      // window will copy them onto the muscles.
      controller.stageExcitations (excitations);

      // First call only: stop the still-free-running Player. Subsequent
      // calls find the player already stopped (no-op).
      s.pause();

      double stepSize = controller.getMechStepSize();
      double endTime  = s.getTime() + subStepsPerAction * stepSize;
      s.playRequest (endTime);

      // Spin until the Player reaches endTime. parkNanos gives finer
      // granularity than Thread.sleep(1) (which Linux rounds up).
      while (s.isPlaying()) {
         java.util.concurrent.locks.LockSupport.parkNanos (50_000); // 50 µs
      }
      return controller.getState();
   }

   private static Scheduler getScheduler() {
      Main m = Main.getMain();
      return (m != null) ? m.getScheduler() : null;
   }
}
