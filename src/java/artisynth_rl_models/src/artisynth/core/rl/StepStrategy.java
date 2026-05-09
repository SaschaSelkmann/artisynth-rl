package artisynth.core.rl;

import java.util.ArrayList;

/**
 * Pluggable policy that controls how the simulation advances between
 * agent actions.
 *
 * <p>The framework historically ran the simulation freely (unthrottled in
 * headless mode) regardless of when the agent submitted excitations. The
 * resulting variable number of sub-steps per RL step makes the environment
 * non-stationary and complicates reward shaping. {@code StepStrategy} lifts
 * that policy out of {@link RlController} so it can be swapped without
 * touching the controller core.
 *
 * <p>Implementations live in this package and are registered on a controller
 * via {@link RlController#setStepStrategy(StepStrategy)}. Two strategies are
 * provided:
 * <ul>
 *   <li>{@link FreeRunStrategy} — legacy behaviour: HTTP thread merely flags
 *       new excitations, the simulation thread picks them up at the next
 *       {@code apply()}; sim runs continuously.</li>
 *   <li>{@link SteppedStrategy} — synchronous: the simulation is paused, then
 *       advanced for a fixed number of sub-steps per submitted action, then
 *       paused again. Each agent step corresponds to a deterministic amount
 *       of simulation time.</li>
 * </ul>
 *
 * <p>Future strategies (real-time wall-clock pacing, convergence-bounded
 * advance, event-triggered stops) plug in by implementing this interface.
 */
public interface StepStrategy {

   /** Short identifier (used in logs and as the CLI argument value). */
   String name();

   /**
    * Called once after the strategy has been registered on a controller and
    * the model is fully built. Implementations may use this hook to set up
    * scheduler state (e.g., put the scheduler into the paused state expected
    * by the strategy).
    */
   default void initialize (RlController controller) {
      // default: no-op
   }

   /**
    * Apply the agent's excitations and advance the simulation according to
    * the strategy's policy. Called from the HTTP handler thread.
    *
    * @param controller the controller invoking the strategy; provides access
    *                   to the mech system and exciter management.
    * @param excitations the excitation vector submitted by the agent.
    * @return state to send back to the agent. May be empty (legacy
    *         behaviour) — in that case the agent is expected to issue a
    *         separate {@code GET /state}.
    */
   RlState applyExcitations (
      RlController controller, ArrayList<Double> excitations);

   /**
    * Whether the simulator is paused between agent actions in this strategy.
    * If {@code true}, the controller's reset path may mutate the model
    * directly on the HTTP thread instead of dispatching through the
    * simulation thread.
    */
   default boolean isSynchronous() {
      return false;
   }

   /**
    * Notify the strategy that the model has just been reset (called after
    * {@code resetState()} so internal counters can be cleared). Default is
    * a no-op.
    */
   default void onReset (RlController controller) {
      // default: no-op
   }
}
