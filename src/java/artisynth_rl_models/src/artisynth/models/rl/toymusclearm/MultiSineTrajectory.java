package artisynth.models.rl.toymusclearm;

import java.util.Random;

/**
 * Reference trajectory built as a sum of two sinusoids per joint:
 *
 * <pre>
 *   theta_i(t) = A1_i sin(2*pi*f1_i t + p1_i) + A2_i sin(2*pi*f2_i t + p2_i)
 * </pre>
 *
 * The two components beat against each other so the apparent amplitude and
 * phase drift naturally over an episode — visually the signal looks like a
 * sin/cos with slowly time-varying parameters, while remaining bounded and
 * differentiable.
 *
 * <p>On {@link #randomize(Random)} all six per-joint parameters are
 * resampled. Sampling ranges are configurable; defaults keep the resulting
 * peak amplitude well inside the toy-arm's hinge limits (±70° / ±120°), so
 * the agent never gets a target that is physically unreachable.
 */
public class MultiSineTrajectory implements JointAngleTrajectory {

   private static final double TWO_PI = 2.0 * Math.PI;

   // Per-joint amplitude envelope bounds (rad). Total amplitude per joint is
   // |A1| + |A2| ∈ [ampMin, ampMax].
   private final double[] ampMin = new double[2];
   private final double[] ampMax = new double[2];

   // Frequency bands shared across joints (Hz).
   private double slowFreqMin = 0.10;
   private double slowFreqMax = 0.25;
   private double fastFreqMin = 0.30;
   private double fastFreqMax = 0.70;

   // Mix between the two components: A1 = mix * total, A2 = (1-mix) * total.
   // Range chosen so the slow component stays dominant.
   private double mixMin = 0.5;
   private double mixMax = 0.8;

   // Per-joint sampled state.
   private final double[] A1 = new double[2];
   private final double[] A2 = new double[2];
   private final double[] f1 = new double[2];
   private final double[] f2 = new double[2];
   private final double[] p1 = new double[2];
   private final double[] p2 = new double[2];

   private boolean randomizeEnabled = true;

   public MultiSineTrajectory() {
      // Joint 0 hinge limit ±70° → keep peak amplitude ≤ 50°.
      ampMin[0] = Math.toRadians (25);  ampMax[0] = Math.toRadians (50);
      // Joint 1 hinge limit ±120° → keep peak amplitude ≤ 80°.
      ampMin[1] = Math.toRadians (35);  ampMax[1] = Math.toRadians (80);

      // Initial values (replaced by first randomize() if enabled).
      A1[0] = Math.toRadians (25); A2[0] = Math.toRadians (15);
      A1[1] = Math.toRadians (40); A2[1] = Math.toRadians (20);
      f1[0] = 0.15; f2[0] = 0.45;
      f1[1] = 0.18; f2[1] = 0.50;
      p1[0] = 0;            p2[0] = 0;
      p1[1] = Math.PI / 2;  p2[1] = 0;
   }

   public void setRandomizeEnabled (boolean enabled) {
      this.randomizeEnabled = enabled;
   }

   public boolean isRandomizeEnabled() {
      return randomizeEnabled;
   }

   public void setAmpRange (int joint, double minRad, double maxRad) {
      ampMin[joint] = minRad;  ampMax[joint] = maxRad;
   }

   public void setSlowFreqRange (double minHz, double maxHz) {
      slowFreqMin = minHz;  slowFreqMax = maxHz;
   }

   public void setFastFreqRange (double minHz, double maxHz) {
      fastFreqMin = minHz;  fastFreqMax = maxHz;
   }

   public void setMixRange (double min, double max) {
      mixMin = min;  mixMax = max;
   }

   @Override
   public double[] angles (double t) {
      return new double[] {
         A1[0] * Math.sin (TWO_PI * f1[0] * t + p1[0]) +
         A2[0] * Math.sin (TWO_PI * f2[0] * t + p2[0]),
         A1[1] * Math.sin (TWO_PI * f1[1] * t + p1[1]) +
         A2[1] * Math.sin (TWO_PI * f2[1] * t + p2[1]),
      };
   }

   @Override
   public double[] velocities (double t) {
      return new double[] {
         A1[0] * TWO_PI * f1[0] * Math.cos (TWO_PI * f1[0] * t + p1[0]) +
         A2[0] * TWO_PI * f2[0] * Math.cos (TWO_PI * f2[0] * t + p2[0]),
         A1[1] * TWO_PI * f1[1] * Math.cos (TWO_PI * f1[1] * t + p1[1]) +
         A2[1] * TWO_PI * f2[1] * Math.cos (TWO_PI * f2[1] * t + p2[1]),
      };
   }

   @Override
   public void randomize (Random rng) {
      if (!randomizeEnabled) {
         return;
      }
      for (int j = 0; j < 2; j++) {
         double total = sample (rng, ampMin[j], ampMax[j]);
         double mix   = sample (rng, mixMin, mixMax);
         A1[j] = total * mix;
         A2[j] = total * (1.0 - mix);
         f1[j] = sample (rng, slowFreqMin, slowFreqMax);
         f2[j] = sample (rng, fastFreqMin, fastFreqMax);
         p1[j] = rng.nextDouble() * TWO_PI;
         p2[j] = rng.nextDouble() * TWO_PI;
      }
   }

   private static double sample (Random rng, double lo, double hi) {
      return lo + rng.nextDouble() * (hi - lo);
   }
}
