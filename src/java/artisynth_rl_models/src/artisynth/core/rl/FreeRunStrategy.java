package artisynth.core.rl;

import java.util.ArrayList;

/**
 * Legacy strategy: HTTP handler stages the excitation vector and returns
 * immediately. The simulation, which runs continuously, picks the new
 * values up the next time {@link RlController#apply(double, double)} is
 * called.
 *
 * <p>This matches the behaviour of the framework before
 * {@link StepStrategy} was introduced. All older demos
 * ({@code RlPoint2PointDemo}, {@code RlLumbarSpineDemo}, {@code RlJawDemo})
 * stay on this strategy so previously trained policies remain valid.
 */
public class FreeRunStrategy implements StepStrategy {

   public static final String NAME = "free_run";

   @Override
   public String name() {
      return NAME;
   }

   @Override
   public RlState applyExcitations (
      RlController controller, ArrayList<Double> excitations) {
      controller.stageExcitations (excitations);
      return new RlState();
   }
}
