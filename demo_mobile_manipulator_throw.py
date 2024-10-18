import time
import math
import pickle
import argparse
import numpy as np
import pybullet as p
import pybullet_data

from sys import path
from pathlib import Path
from ruckig import InputParameter, Ruckig, Trajectory, Result

# Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))

def plan_and_simulate_throw(box_position):
    """
    Plan and simulate a throw based on the given target box position.
    
    :param box_position: The position of the target box in the simulation environment.
    """
    z_target = box_position[2]
    base_start_position = -box_position[:2]

    # Initial joint configuration of the robot
    joint_start_position = np.array([0.0, -np.pi / 4, 0.0, -np.pi, 0.0, np.pi * 3 / 4, np.pi / 4])
    joint_start_velocity = np.zeros(7)

    robot_data_path = "robot_data/panda_5_joint_dense_1_dataset_15"
    experiment_data_path = "object_data/brt_gravity_only"
    gravity = -9.81

    # Connect to PyBullet in DIRECT mode (no GUI)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    urdf_path = "franka_panda/panda.urdf"
    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Get initial guesses for joint configurations, phi angles, and throw positions
    q_candidates, phi_candidates, throw_candidates = match_robot_brt_data(z_target, robot_data_path, experiment_data_path)
    
    # Ensure throw_candidates is a NumPy array for multidimensional indexing
    throw_candidates = np.array(throw_candidates)
    
    num_candidates = len(q_candidates)
    
    # Store valid throw configurations and their corresponding trajectories and durations
    traj_durations = []
    trajs = []
    valid_throw_configs = []

    start_time = time.time()

    # Loop through each candidate and calculate the throw trajectory
    for i in range(num_candidates):
        throw_config = get_throw_config(robot, q_candidates[i], phi_candidates[i], throw_candidates[i])

        # Filter out invalid configurations that would result in a collision with the gripper palm
        if throw_config[4][2] < -0.02:
            continue

        # Generate the throwing trajectory using Ruckig
        traj_throw = get_traj_from_ruckig(
            joint_start_position=joint_start_position,
            joint_start_velocity=joint_start_velocity,
            joint_throw_position=throw_config[0],
            joint_throw_velocity=throw_config[3],
            base_start_position=base_start_position,
            based=-throw_config[-1][:-1]
        )

        traj_durations.append(traj_throw.duration)
        trajs.append(traj_throw)
        valid_throw_configs.append(throw_config)

    elapsed_time = time.time() - start_time
    print(f"Given query z={z_target:.2f}, found {len(valid_throw_configs)} valid throws in {elapsed_time * 1000:.2f} ms")

    # Select the minimum-time trajectory for simulation
    selected_idx = np.argmin(traj_durations)
    traj_throw = trajs[selected_idx]
    selected_throw_config = valid_throw_configs[selected_idx]

    # Disconnect from PyBullet
    p.disconnect()

    # Output information about the selected throw
    print("box_position: ", selected_throw_config[-1])
    print(f"throwing range: {-throw_candidates[selected_idx, 0]:0.2f}, throwing height: {throw_candidates[selected_idx, 1]:0.2f}")
    
    # Simulate the selected throw
    throw_simulation_mobile(traj_throw, selected_throw_config, gravity)

def match_robot_brt_data(z_target_to_base, robot_data_path, brt_data_path, velocity_threshold=0.1):
    """
    Match robot data with BRT data to find initial guesses of (joint_angles, phi, positions) for trajectory planning.
    
    :param z_target_to_base: Target z-position relative to the robot base.
    :param robot_data_path: Path to the robot's configuration data files.
    :param brt_data_path: Path to the BRT data files.
    :param velocity_threshold: Threshold to filter out invalid velocities.
    :return: Lists of matched joint configurations, phi angles, and target positions.
    """
    start_time = time.time()

    phi_angles = np.linspace(-90, 90, 13)  # Possible phi values to check
    robot_z_levels = np.arange(0.0, 1.11, 0.05)  # Discretized z-levels for the robot arm
    num_robot_z_levels = len(robot_z_levels)

    # Load robot and BRT data
    joint_mesh = np.load(f"{robot_data_path}/qs.npy")
    robot_velocity_data = np.load(f"{robot_data_path}/phi_gamma_velos_naive.npy")
    joint_indices = np.load(f"{robot_data_path}/phi_gamma_q_idxs_naive.npy")

    brt_z_levels = np.load(f"{brt_data_path}/brt_zs.npy")
    brt_min_z = np.min(brt_z_levels)
    shift_index = round((z_target_to_base + brt_min_z) / 0.05)

    with open(f"{brt_data_path}/brt_chunk.pkl", 'rb') as fp:
        brt_data_chunks = pickle.load(fp)

    matched_joint_angles, matched_phi_angles, matched_positions = [], [], []

    for i, robot_z in enumerate(robot_z_levels):
        if i - shift_index >= len(brt_z_levels):
            continue
        
        for gamma_index in range(robot_velocity_data.shape[2]):
            brt_data_for_level = brt_data_chunks[i - shift_index][gamma_index]
            if brt_data_for_level is None:
                continue

            for phi_index, phi_angle in enumerate(phi_angles):
                max_velocity = robot_velocity_data[i, phi_index, gamma_index]
                valid_candidates = brt_data_for_level[brt_data_for_level[:, 4] < max_velocity - velocity_threshold]

                if valid_candidates.shape[0] > 0:
                    num_candidates = valid_candidates.shape[0]
                    matched_joint_angles += [joint_mesh[joint_indices[i, phi_index, gamma_index].astype(int), :].flatten()] * num_candidates
                    matched_phi_angles += [phi_angle] * num_candidates
                    matched_positions += list(valid_candidates[:, :-1])

    elapsed_time = time.time() - start_time
    print(f"Query z={z_target_to_base:.2f}, found {len(matched_joint_angles)} candidates in {elapsed_time * 1000:.2f} ms")

    return matched_joint_angles, matched_phi_angles, matched_positions

def get_throw_config(robot, q, phi, throw):
    """
    Return full throwing configurations.
    
    :param robot: The robot model in the simulation.
    :param q: Current joint positions of the robot.
    :param phi: Throwing angle (in degrees).
    :param throw: Throw parameters [r_throw, z_throw, r_dot, z_dot].
    :return: Updated joint positions, phi, throw, joint velocities, block position, end-effector velocity, 
             end-effector position (AE), and the box position.
    """
    r_throw, z_throw, r_dot, z_dot = throw

    # Update robot's joint states
    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
    p.resetJointStatesMultiDof(robot, controlled_joints, [[pos] for pos in q])
    
    # Get the end-effector position
    end_effector_pos = p.getLinkState(robot, 11)[0]
    
    # Compute Jacobian for joint velocities
    J, _ = p.calculateJacobian(robot, 11, [0, 0, 0], q.tolist() + [0.1, 0.1], [0.0] * 9, [0.0] * 9)
    J = np.array(J)[:3, :7]
    J_inv = np.linalg.pinv(J)
    
    # Calculate throwing angle and end-effector velocity
    throwing_angle = np.arctan2(end_effector_pos[1], end_effector_pos[0]) + np.deg2rad(phi)
    velocity_direction = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])
    
    end_effector_velocity = np.array([velocity_direction[0] * r_dot, velocity_direction[1] * r_dot, z_dot])
    joint_velocities = J_inv @ end_effector_velocity
    
    # Compute the box position based on the throw
    box_position = end_effector_pos + np.array([-r_throw * velocity_direction[0], 
                                                -r_throw * velocity_direction[1], 
                                                -z_throw])

    # Adjust the final joint angle to avoid hitting the gripper
    gripper_pos, gripper_orn = p.getLinkState(robot, 11)[0:2]
    inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
    
    adjusted_end_effector_velocity_dir = end_effector_velocity / np.linalg.norm(end_effector_velocity)
    temp_pos = end_effector_pos + adjusted_end_effector_velocity_dir
    block_pos_in_gripper, _ = p.multiplyTransforms(inv_gripper_pos, inv_gripper_orn, temp_pos, [0, 0, 0, 1])
    
    velocity_angle_in_gripper = np.arctan2(block_pos_in_gripper[1], block_pos_in_gripper[0])

    if abs(velocity_angle_in_gripper) < 0.5 * np.pi:
        adjusted_angle = velocity_angle_in_gripper
    else:
        adjusted_angle = velocity_angle_in_gripper - np.sign(velocity_angle_in_gripper) * np.pi

    q[-1] = adjusted_angle

    return q, phi, throw, joint_velocities, block_pos_in_gripper, end_effector_velocity, end_effector_pos, box_position

def get_traj_from_ruckig(joint_start_position, joint_start_velocity, joint_throw_position, joint_throw_velocity, base_start_position, based):
    inp = InputParameter(9)
    inp.current_position = np.concatenate((joint_start_position, base_start_position))
    inp.current_velocity = np.concatenate((joint_start_velocity, np.zeros(2)))
    inp.current_acceleration = np.zeros(9)

    inp.target_position = np.concatenate((joint_throw_position, based))
    inp.target_velocity = np.concatenate((joint_throw_velocity, np.zeros(2)))
    inp.target_acceleration = np.zeros(9)

    inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 2.0, 2.0]) * 10
    inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20, 5.0, 5.0]) -1.0
    inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000, 1000, 1000]) - 100

    otg = Ruckig(9)
    trajectory = Trajectory(9)
    _ = otg.calculate(inp, trajectory)

    return trajectory

def throw_simulation_mobile(trajectory, throw_config, gravity=-9.81):
    """
    Simulate the robot's throwing motion in PyBullet with the given trajectory and configuration.
    
    :param trajectory: The throwing trajectory generated by Ruckig.
    :param throw_config: The configuration of the throw including joint states and object positions.
    :param gravity: The gravity value for the simulation (default: -9.81).
    """
    PANDA_BASE_HEIGHT = 0.5076438625
    box_position = throw_config[-1]
    
    # Connect to PyBullet GUI
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40, cameraTargetPosition=[0.75, -0.75, 0])

    # Simulation parameters
    freq = 1000
    dt = 1.0 / freq
    p.setGravity(0, 0, gravity)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)

    controlled_joints = [3, 4, 5, 6, 7, 8, 9]
    gripper_joints = [12, 13]

    # Load robot and environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf", [-box_position[0], -box_position[1], 0], useFixedBase=True)

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    soccerball_id = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    box_id = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                        [0, 0, PANDA_BASE_HEIGHT + box_position[2]], globalScaling=0.5)

    # Configure dynamics for objects
    p.changeDynamics(soccerball_id, -1, mass=1.0, rollingFriction=0.03, spinningFriction=0.03, restitution=0.2)
    p.changeDynamics(plane_id, -1, restitution=0.9)
    p.changeDynamics(robot_id, gripper_joints[0], jointUpperLimit=100)
    p.changeDynamics(robot_id, gripper_joints[1], jointUpperLimit=100)

    # Prepare for trajectory execution
    start_time, end_time = 0, trajectory.duration
    plan_time = end_time - start_time
    sample_time = np.arange(0, end_time, dt)
    n_steps = sample_time.shape[0]
    traj_data = np.zeros([3, n_steps, 7])
    base_traj_data = np.zeros([3, n_steps, 2])
    for i in range(n_steps):
        for j in range(3):
            tmp = trajectory.at_time(sample_time[i])[j]
            traj_data[j, i] = tmp[:7]
            base_traj_data[j, i] = tmp[-2:]

    # Reset robot and objects to initial positions
    joint_start_position = traj_data[0, 0]
    p.resetBasePositionAndOrientation(robot_id, np.append(base_traj_data[0,0], 0.0), [0, 0, 0, 1])
    p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in joint_start_position])

    end_effector_state = p.getLinkState(robot_id, 14, computeLinkVelocity=1)
    p.resetBasePositionAndOrientation(soccerball_id, end_effector_state[0], [0, 0, 0, 1])
    p.resetJointState(robot_id, gripper_joints[0], 0.03)
    p.resetJointState(robot_id, gripper_joints[1], 0.03)
    
    current_time = 0
    is_throw_active = True
    
    while(True):
        if is_throw_active:
            ref_trajectory = trajectory.at_time(current_time)
            ref_joints = [ref_trajectory[i][:7] for i in range(3)]
            ref_base = [ref_trajectory[i][-2:] for i in range(3)]
            
            # Update joint states and base position
            p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in ref_joints [0]], targetVelocities=[[joint_start_position_i] for joint_start_position_i in ref_joints [1]])
            p.resetBasePositionAndOrientation(robot_id, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        else:
            ref_trajectory = trajectory.at_time(plan_time)
            ref_joints = [ref_trajectory[i][:7] for i in range(3)]
            ref_base = [ref_trajectory[i][-2:] for i in range(3)]

            p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in ref_joints [0]])
            p.resetBasePositionAndOrientation(robot_id, np.append(ref_base[0], 0.0), [0, 0, 0, 1])

        # Control the gripper state
        if current_time > plan_time - 1*dt:
            p.resetJointState(robot_id, gripper_joints[0], 0.05)
            p.resetJointState(robot_id, gripper_joints[1], 0.05)
        else:
            end_effector_state = p.getLinkState(robot_id, 14, computeLinkVelocity=1)
            p.resetBasePositionAndOrientation(soccerball_id, end_effector_state[0], [0, 0, 0, 1])
            p.resetBaseVelocity(soccerball_id, linearVelocity=end_effector_state[-2])
        p.stepSimulation()
        current_time = current_time + dt
        if current_time > trajectory.duration:
            is_throw_active = False
        time.sleep(dt)
        if current_time > 5.0:
            break
    
    # Disconnect from simulation
    p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Overall
    box_position = np.array([1.0, 0.0, 0.0])
    print(f"box position in panda frame: {box_position}")
    
    plan_and_simulate_throw(box_position)