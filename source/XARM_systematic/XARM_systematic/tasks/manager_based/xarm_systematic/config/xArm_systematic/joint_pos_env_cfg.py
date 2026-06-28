# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

##
# Pre-defined configs
##
# Assuming your xarm config is in a file named 'xarm.py' relative to this one
from .xarm import XARM5_CFG


##
# Environment configuration
##

@configclass
class XArm5CommandsCfg:
    """Command terms for the MDP."""
    #
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link6",  # Target the end-effector link of xArm6
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.6),  # Adjusted workspace for xArm reach
            pos_y=(-0.4, 0.4),  # +/- 30cm side to side
            pos_z=(0.12, 0.6),  # Height range
            roll=(0.0, 0.0),  # Keep orientation fixed for simple reaching
            pitch=(math.pi, math.pi),  # Pointing down (or adjust as needed)
            yaw=(-math.pi, math.pi),
        ),
    )

    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="link6",  # Target the end-effector link of xArm6
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.2535, 0.2535),  # Adjusted workspace for xArm reach
    #         pos_y=(0, 0),  # +/- 30cm side to side
    #         pos_z=(0.195, 0.195),  # Height range
    #         roll=(0.0, 0.0),  # Keep orientation fixed for simple reaching
    #         pitch=(math.pi, math.pi),  # Pointing down (or adjust as needed)
    #         yaw=(math.pi, math.pi),
    #     ),
    # )


@configclass
class XArm5CurriculumCfg:
    """Curriculum terms for the MDP."""
    # pass
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


@configclass
class XArm5ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"],
                                                preserve_order=True)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"],
                                                preserve_order=True)}
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class XArm6RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task Terms --
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "command_name": "ee_pose"},
    )

    # -- Penalties --
    # COMPLETELY REMOVE action_l2. It conflicts with relative actions.
    # action = RewTerm(func=mdp.action_l2, weight=-0.0005)
    # Keep these very small to prevent jitter, but not so large that they paralyze the arm
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalize hitting joint limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.02)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)


@configclass
class XARMReachEnvCfg(ReachEnvCfg):
    observations: XArm5ObservationsCfg = XArm5ObservationsCfg()
    commands: XArm5CommandsCfg = XArm5CommandsCfg()
    rewards: XArm6RewardsCfg = XArm6RewardsCfg()

    curriculum: XArm5CurriculumCfg = XArm5CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 1. Switch robot to xArm6
        self.scene.robot = XARM5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. General Environment Settings
        self.scene.env_spacing = 2.0
        self.episode_length_s = 100.0

        # 3. Override Actions
        # Joint 4 is locked in the asset config, so we explicitly EXCLUDE it from actions.


        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"],
        #     preserve_order=True,
        #     scale=1,
        #     debug_vis=True,
        #     use_default_offset=False,
        #     clip={".*": (-math.pi, math.pi)}
        # )

        # 3. Override Actions
        # Schimbăm la control Relativ (Delta) pentru o tranziție mult mai sigură pe hardware-ul fizic
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            # TREBUIE să excludem din nou joint4! Nu folosi ".*"
            joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"],
            preserve_order=True,
            scale=0.0625,  # ~3.6 degrees per action step (foarte sigur) 0.0625
            use_zero_offset=True,
        )

        # 4. Override Rewards & Commands body names
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["EF"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["EF"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["EF"]

        self.commands.ee_pose.body_name = "EF"

        # --- NOU: SISTEMUL DE PREVENȚIE A COLIZIUNILOR (KILL SWITCH) ---

        # 1. Adăugăm un senzor de contact pe robot
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            # Această linie îi spune senzorului să raporteze doar dacă robotul se atinge de EL ÎNSUȘI
            # Dacă ai o masă, poți adăuga și masa aici: ["{ENV_REGEX_NS}/Robot/.*", "{ENV_REGEX_NS}/Table"]
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]
        )

        # 2. Regulă: Oprește episodul dacă senzorul detectează forță (coliziune)
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0},
        )

        # 3. Randomizare mai sigură la pornire (evităm spawnarea în coliziune)
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"]),
                # Un range de +/- 0.5 radiani (aprox 28 grade) previne formarea unui "covrig" la resetare
                "position_range": (-0.5, .5),
                "velocity_range": (0.0, 0.0),
            },
        )

        # 5. DOMAIN RANDOMIZATION (The Pro Sim-to-Real Fix)
        # Randomize joint starting positions slightly
        # self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)  # Small noise on spawn

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"]),
                "position_range": (-math.pi/2, math.pi/2),
                "velocity_range": (0.0, 0.0),
            },
        )

        # Randomize Joint Friction and Damping
        # Antrenează agentul să controleze brațul indiferent dacă hardware-ul se mișcă greoi sau foarte rapid.
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                # Make the ranges much tighter (less chaotic)
                "stiffness_distribution_params": (0.96, 1.04),  # 95-105%
                "damping_distribution_params": (0.91, 1.09),  # 90-110%
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "friction_distribution_params": (0.0, 0.05),  # Add 0-0.1 Nm friction
                "operation": "add",
                "distribution": "uniform",
            },
        )

        # Randomize Link Masses
        # Variază inerția teoretică a brațului.
        self.events.randomize_rigid_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),  # Randomize mass +/- 20%
                "operation": "scale",
            },
        )


@configclass
class XARMReachEnvCfg_PLAY(XARMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play so you see pure policy execution
        self.observations.policy.enable_corruption = False