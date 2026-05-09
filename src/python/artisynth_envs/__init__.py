from common import constants as c
from gymnasium.envs.registration import register

register(
    id='Point2PointEnv-v1',
    entry_point='artisynth_envs.envs:Point2PointEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.point2point.RlPoint2PointDemo',
            'artisynth_args': '-demoType 1d -muscleOptLen 0.1 -radius 8',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'point'}],
                c.TARGET: [{c.NAME: 'point_ref'}],
                c.PROPS: ['position']}
            }
)

register(
    id='Point2PointEnv-v2',
    entry_point='artisynth_envs.envs:Point2PointEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.point2point.RlPoint2PointDemo',
            'artisynth_args': '-num 8 -demoType 2d -muscleOptLen 0.1 -radius 5',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'point'}],
                c.TARGET: [{c.NAME: 'point_ref'}],
                c.PROPS: ['position']}
            }
)

register(
    id='Point2PointEnv-v3',
    entry_point='artisynth_envs.envs:Point2PointEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.point2point.RlPoint2PointDemo',
            'artisynth_args': '-demoType 3d -muscleOptLen 0.1 -radius 5',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'point'}],
                c.TARGET: [{c.NAME: 'point_ref'}],
                c.PROPS: ['position']}
            }
)

register(
    id='Point2PointEnv-v4',
    entry_point='artisynth_envs.envs:Point2PointEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.point2point.RlPoint2PointDemo',
            'artisynth_args': '-demoType nonSym -muscleOptLen 0.1 -radius 5',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'point'}],
                c.TARGET: [{c.NAME: 'point_ref'}],
                c.PROPS: ['position']}
            }
)

register(
    id='SpineEnv-v0',
    entry_point='artisynth_envs.envs:SpineEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.lumbarspine.RlLumbarSpineDemo',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'thorax',   c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L1',        c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L2',        c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L3',        c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L4',        c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L5',        c.LOW: [-1.]*4, c.HIGH: [1.]*4}],
                c.TARGET:  [{c.NAME: 'thorax_ref', c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L1_ref',    c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L2_ref',    c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L3_ref',    c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L4_ref',    c.LOW: [-1.]*4, c.HIGH: [1.]*4},
                            {c.NAME: 'L5_ref',    c.LOW: [-1.]*4, c.HIGH: [1.]*4}],
                c.PROPS: ['orientation']}
            }
)

register(
    id='JawEnv-v0',
    entry_point='artisynth_envs.envs:JawEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.jaw.RlJawDemo',
            'artisynth_args': '-disc false -condyleConstraints true -condylarCapsule false',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'lowerincisor',
                             c.LOW: [-7, -105, -80], c.HIGH: [14, -80, -35]}],
                c.TARGET:  [{c.NAME: 'lowerincisor_ref',
                             c.LOW: [-7, -105, -80], c.HIGH: [14, -80, -35]}],
                c.PROPS: ['position']},
            }
)

register(
    id='JawEnv-v1',
    entry_point='artisynth_envs.envs:JawEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.jaw.RlJawDemo',
            'artisynth_args': '-disc false -condyleConstraints false -condylarCapsule true',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'lowerincisor',
                             c.LOW: [-7, -105, -80], c.HIGH: [14, -80, -35]}],
                c.TARGET:  [{c.NAME: 'lowerincisor_ref',
                             c.LOW: [-7, -105, -80], c.HIGH: [14, -80, -35]}],
                c.PROPS: ['position']},
            }
)

register(
    id='JawEnv-v2',
    entry_point='artisynth_envs.envs:JawEnv',
    kwargs={'artisynth_model': 'artisynth.models.rl.jaw.RlJawDemo',
            'artisynth_args': '-disc false -condyleConstraints false -condylarCapsule true',
            c.COMPONENTS: {
                c.CURRENT: [{c.NAME: 'lowerincisor',
                             c.LOW: [-7, -105, -80, -10, -10, -10],
                             c.HIGH: [14, -80, -35, 10, 10, 10]}],
                c.TARGET:  [{c.NAME: 'lowerincisor_ref',
                             c.LOW: [-7, -105, -80, -10, -10, -10],
                             c.HIGH: [14, -80, -35, 10, 10, 10]}],
                c.PROPS: ['position', 'velocity']},
            }
)

# Two-link arm with four muscles tracking a sin/cos joint-angle trajectory.
# RlProps (joint0/joint1 + *_ref) replace the `components` dict-based observation
# layout used by the other envs, so no `components` kwargs are needed here.
register(
    id='ToyMuscleArmEnv-v0',
    entry_point='artisynth_envs.envs:ToyMuscleArmEnv',
    kwargs={
        'artisynth_model': 'artisynth.models.rl.toymusclearm.RlToyMuscleArmDemo',
        # Default: deterministic stepped advance + multisine reference. The
        # `-waitAction` value must match the Python `wait_action` so each
        # agent step covers the same sim time.
        'artisynth_args': '-episodeDuration 10 -randomizeTraj true '
                           '-stepStrategy stepped -waitAction 0.05 '
                           '-trajKind multisine',
        # `components` is required by ArtiSynthBase but unused here — the env
        # reads RlProps directly via its own state_dic_to_array override.
        c.COMPONENTS: {c.CURRENT: [], c.TARGET: [], c.PROPS: []},
        'episode_duration': 10.0,
        'wait_action': 0.05,
        'goal_threshold_deg': 5.0,
        'goal_reward': 0.1,
        'w_u': 1.0,
        'w_r': 0.01,
        # Pure physical-failure termination (no tracking-error checks).
        'pinned_enabled': True,
        'pinned_angle_tol_deg': 1.0,
        'pinned_velocity_tol': 0.1,
        'pinned_consecutive_steps': 20,           # 1 s sim time
        'velocity_blowup_enabled': True,
        'velocity_blowup_rad_s': 30.0,
        'velocity_blowup_consecutive_steps': 5,
    }
)
