# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UFACTORY xArm6.

The following configuration parameters are available:

* :obj:`XARM6_CFG`: The xArm6 robot arm with detailed actuator limits.

Reference: https://github.com/xArm-Developer/xarm_ros
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math

##
# Configuration
##
XARM5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # TODO: Update this path to the actual location of your generated USD file
        usd_path="../X_arm/xarm_systematic/xarm_systematic.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # <-- FIX 1: Dezactivăm gravitația pentru a preveni deviațiile și tremurul
            max_depenetration_velocity=5.0,
            #
            # max_linear_velocity=5.0,
            # max_angular_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Initial joint positions (radians)
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #         # Initial joint positions as ranges (min, max) in radians
    #         # This gives the environment the boundaries for randomization
    #         joint_pos={
    #             "joint1": (-math.pi/2, math.pi/2),  # Base rotation +/- 90 degrees
    #             "joint2": (-1.0, 1.0),              # Shoulder tilt limits
    #             "joint3": (-1.0, 1.0),              # Elbow tilt limits
    #             "joint4": (0.0, 0.0),               # Locked joint stays exactly at 0
    #             "joint5": (-math.pi/2, math.pi/2),  # Wrist tilt limits
    #             "joint6": (-math.pi, math.pi),      # Wrist rotation full circle
    #         },
    #     ),
    actuators={
        # Group 1: Base and Shoulder (Joints 1 & 2)
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint1","joint2"],
            effort_limit_sim=50.0,
            velocity_limit=3.14/2,
            # FIX 2 & 3: Aplicăm regula "ordinului de mărime" (10:1 ratio)
            stiffness=400.0,
            damping=20.0,
            friction=0.0,         # <-- FIX 3: Frecare zero pentru a nu amplifica oscilațiile
        ),
        # Group 2: Elbow and Wrist 1 (Joints 3 & 5 ONLY - Excludes 4)
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint3","joint5"],
            effort_limit_sim=32.0,
            velocity_limit=3.14/2,
            stiffness=400.0,
            damping=20.0,
            friction=0.0,
        ),
        # Group 3: Wrist 2 (Joint 6)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit_sim=20.0,
            velocity_limit=3.14/2,
            # End-effector-ul trebuie să fie puțin mai flexibil pentru a nu vibra pe loc
            stiffness=400.0,
            damping=20.0,
            friction=0.0,
        ),
        # Group 4: LOCKED Joint 4
        "fixed_joint": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=32.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)
"""Configuration of xArm6 with Joint 4 structurally locked."""