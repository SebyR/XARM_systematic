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

##
# Pre-defined configs
##
# Assuming your xarm6 config is in a file named 'xarm6.py' relative to this one
from .xarm import XARM5_CFG


##
# Environment configuration
##

@configclass
class XArm5CommandsCfg:
    """Command terms for the MDP."""

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
    #         pos_x=(0.2535, 0.2536),  # Adjusted workspace for xArm reach
    #         pos_y=(0, 0),  # +/- 30cm side to side
    #         pos_z=(0.095, 0.095),  # Height range
    #         roll=(0.0, 0.0),  # Keep orientation fixed for simple reaching
    #         pitch=(math.pi, math.pi),  # Pointing down (or adjust as needed)
    #         yaw=(math.pi, math.pi),
    #     ),
    # )


@configclass
class XArm5CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )
    #
    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )


@configclass
class XArm5ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Observation terms (order preserved)
        # We explicitly list joints to exclude the locked 'joint4' if desired,
        # or include all if you want the policy to know about the locked state.
        # Here we observe ALL joints so the policy knows the full state.
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"], preserve_order=True)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"], preserve_order=True)}
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

#OLD BUT GOLD REWARD
# @configclass
# class XArm6RewardsCfg:
#     """Reward terms for the MDP."""
#
#     # -- Task Terms --
#     end_effector_position_tracking = RewTerm(
#         func=mdp.position_command_error,
#         weight=-0.2,
#         params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"]), "command_name": "ee_pose"},
#     )
#     end_effector_position_tracking_fine_grained = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.1,
#         params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"]), "std": 0.1, "command_name": "ee_pose"},
#     )
#     end_effector_orientation_tracking = RewTerm(
#         func=mdp.orientation_command_error,
#         weight=-0.1,  # -0.2
#         params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"]), "command_name": "ee_pose"},
#     )
#
#     # -- Penalties --
#     # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
#     # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
#
#     # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
#     # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
#
#
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-0.01,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )
#
#
#     # Penalize hitting joint limits
#     dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.02)

@configclass
class XArm6RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task Terms --
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.2,  # -0.2
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["EF"]), "command_name": "ee_pose"},
    )

    # -- Penalties --
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)


    # Penalize hitting joint limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.02)



@configclass
class XARMReachEnvCfg(ReachEnvCfg):
    observations: XArm5ObservationsCfg = XArm5ObservationsCfg()
    commands: XArm5CommandsCfg = XArm5CommandsCfg()
    rewards: XArm6RewardsCfg = XArm6RewardsCfg()
    # curriculum: XArm5CurriculumCfg = XArm5CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 1. Switch robot to xArm6
        self.scene.robot = XARM5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. General Environment Settings
        self.scene.env_spacing = 2.0
        self.episode_length_s = 300.0

        # 3. Override Actions
        # We only control the moving joints.
        # Joint 4 is locked in the asset config, so we explicitly EXCLUDE it from actions.
        # This prevents the policy from trying to move the locked joint.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            # We explicitly skip joint4 in the action space
            joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"],
            preserve_order=True,
            scale=1,
            debug_vis = True,
            use_default_offset=False,
            clip={".*": (-math.pi, math.pi)}
        )

        # 4. Override Rewards & Commands body names
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["EF"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["EF"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["EF"]
        self.commands.ee_pose.body_name = "EF"

        # 5. Domain Randomization Events (From D1g1tal Ref)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)  # Reset to home or small noise

        # self.events.physics_material = EventTerm(
        #     func=mdp.randomize_rigid_body_material,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        #         "static_friction_range": (0.4, 1.0),
        #         "dynamic_friction_range": (0.4, 0.9),
        #         "restitution_range": (0.0, 0.1),
        #         "num_buckets": 64,
        #     },
        # )
        #
        # self.events.scale_all_link_masses = EventTerm(
        #     func=mdp.randomize_rigid_body_mass,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
        #         "mass_distribution_params": (0.8, 1.2),  # Scale +/- 20%
        #         "operation": "scale"
        #     },
        # )

        # Randomize joint friction
        # self.events.scale_all_joint_friction = EventTerm(
        #     func=mdp.randomize_joint_parameters,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        #         "friction_distribution_params": (0.5, 1.5),
        #         "operation": "scale"
        #     },
        # )


@configclass
class XARMReachEnvCfg_PLAY(XARMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False