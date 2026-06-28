import argparse
from isaaclab.app import AppLauncher

# ==========================================
# 1. Initialize Simulation Application
# ==========================================
parser = argparse.ArgumentParser(description="xArm Sim-to-Real Calibration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import csv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# Se importă configurația ta din xarm.py
from xarm import XARM5_CFG


# ==========================================
# 2. Main Calibration Loop (Simulation)
# ==========================================
def main():
    # Configurăm fizica la 100 Hz, exact cum ai pe robotul fizic (dt = 0.01)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])

    # Spawnăm robotul
    robot_cfg = XARM5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    sim.reset()

    # Numele exacte din Isaac Lab (corespund cu indexul fizic 0, 1, 2, 3, 4)
    joint_names = ["joint1", "joint2", "joint3", "joint5", "joint6"]

    # Parametrii undei sinusoidale (identici cu mediul real)
    amplitude = 0.3  # radiani (~17 grade)
    frequency = 0.5  # Hz
    duration = 4.0  # secunde per motor
    dt = 0.01  # Timp de update simulare (100 Hz)

    # Offset-ul de siguranță pentru Joint 3 (~28 grade / 0.5 radiani)
    JOINT3_OFFSET = -0.5

    data_log = []

    print("Starting sequential simulation test...")

    for joint_name in joint_names:
        print(f"Începem testul simulat pentru {joint_name}...")

        # 1. Definim poziția de bază, integrând offset-ul pentru joint3
        default_pos = torch.zeros((1, robot.num_joints), device=sim.device)
        joint3_idx_sim = robot.find_joints("joint3")[0][0]
        default_pos[0, joint3_idx_sim] = JOINT3_OFFSET

        default_vel = torch.zeros((1, robot.num_joints), device=sim.device)

        # Resetăm simularea curat la poziția de bază sigură la începutul fiecărui motor
        robot.write_joint_state_to_sim(default_pos, default_vel)
        sim.step()

        # Găsim indexul specific pentru articulația testată curent
        active_joint_idx = robot.find_joints(joint_name)[0][0]

        current_time = 0.0

        while current_time <= duration:
            # 2. Calculăm valoarea sinusoidală pură
            sine_wave = amplitude * math.sin(2 * math.pi * frequency * current_time)

            # Menținem toate motoarele la default_pos, modificând DOAR cel activ, PESTE offset
            target_pos = default_pos.clone()
            target_pos[0, active_joint_idx] = default_pos[0, active_joint_idx] + sine_wave

            # 3. Aplicăm comanda (PD control via ImplicitActuatorCfg)
            robot.set_joint_position_target(target_pos)

            # 4. Avansăm fizica simulată
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

            # 5. Extragem valorile pentru logare
            actual_angle = robot.data.joint_pos[0, active_joint_idx].item()
            target_angle_to_log = target_pos[0, active_joint_idx].item()

            data_log.append({
                "joint": joint_name,
                "time": round(current_time, 3),  # Rotunjim pentru un CSV mai curat
                "target_pos": target_angle_to_log,
                "actual_pos": actual_angle
            })

            current_time += dt

    # 6. Salvare în CSV a datelor simulate
    if len(data_log) > 0:
        csv_filename = "sim_xarm_data.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["joint", "time", "target_pos", "actual_pos"])
            writer.writeheader()
            writer.writerows(data_log)
        print(f"Toate datele simulate au fost salvate cu succes în '{csv_filename}'.")

    simulation_app.close()


if __name__ == "__main__":
    main()