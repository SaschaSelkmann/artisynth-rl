package artisynth.models.rl.toymusclearm;

import java.util.Random;

/**
 * Sinusoidal reference trajectory:
 *
 * <pre>
 *   theta_i(t)    =  amp_i * sin(2*pi*freq_i * t + phase_i)
 *   thetaDot_i(t) =  amp_i * 2*pi*freq_i * cos(2*pi*freq_i * t + phase_i)
 * </pre>
 *
 * for {@code i = 0, 1}. Default values produce a smooth sweep within the
 * physical joint limits of the ToyMuscleArm.
 */
public class SinCosTrajectory implements JointAngleTrajectory {

   private static final double TWO_PI = 2.0 * Math.PI;

   private final double[] amp = new double[2];      // rad
   private final double[] freq = new double[2];     // Hz
   private final double[] phase = new double[2];    // rad

   // Sampling ranges used by randomize(); upper bounds keep the targets
   // within the hinge limits (+/-70 deg for joint 0, +/-120 deg for joint 1).
   private double[] ampRange0 = { Math.toRadians(20), Math.toRadians(50) };
   private double[] ampRange1 = { Math.toRadians(30), Math.toRadians(80) };
   private double[] freqRange = { 0.10, 0.30 };
   private boolean randomizeEnabled = true;

   public SinCosTrajectory() {
      amp[0]   = Math.toRadians(35);
      amp[1]   = Math.toRadians(60);
      freq[0]  = 0.15;
      freq[1]  = 0.20;
      phase[0] = 0.0;
      phase[1] = Math.PI / 2.0;
   }

   public SinCosTrajectory (
      double amp0, double amp1, double freq0, double freq1,
      double phase0, double phase1) {
      amp[0] = amp0;       amp[1] = amp1;
      freq[0] = freq0;     freq[1] = freq1;
      phase[0] = phase0;   phase[1] = phase1;
   }

   public void setRandomizeEnabled (boolean enabled) {
      this.randomizeEnabled = enabled;
   }

   public boolean isRandomizeEnabled() {
      return randomizeEnabled;
   }

   public void setAmpRange (int joint, double minRad, double maxRad) {
      double[] r = (joint == 0) ? ampRange0 : ampRange1;
      r[0] = minRad;  r[1] = maxRad;
   }

   public void setFreqRange (double minHz, double maxHz) {
      freqRange[0] = minHz;  freqRange[1] = maxHz;
   }

   @Override
   public double[] angles (double t) {
      return new double[] {
         amp[0] * Math.sin (TWO_PI * freq[0] * t + phase[0]),
         amp[1] * Math.sin (TWO_PI * freq[1] * t + phase[1])
      };
   }

   @Override
   public double[] velocities (double t) {
      return new double[] {
         amp[0] * TWO_PI * freq[0] * Math.cos (TWO_PI * freq[0] * t + phase[0]),
         amp[1] * TWO_PI * freq[1] * Math.cos (TWO_PI * freq[1] * t + phase[1])
      };
   }

   @Override
   public void randomize (Random rng) {
      if (!randomizeEnabled) {
         return;
      }
      amp[0]   = sample (rng, ampRange0[0], ampRange0[1]);
      amp[1]   = sample (rng, ampRange1[0], ampRange1[1]);
      freq[0]  = sample (rng, freqRange[0], freqRange[1]);
      freq[1]  = sample (rng, freqRange[0], freqRange[1]);
      phase[0] = rng.nextDouble() * TWO_PI;
      phase[1] = rng.nextDouble() * TWO_PI;
   }

   private static double sample (Random rng, double lo, double hi) {
      return lo + rng.nextDouble() * (hi - lo);
   }
}
