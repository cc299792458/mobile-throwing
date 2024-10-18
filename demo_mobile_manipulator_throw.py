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
    # Height of target box relative to panda base, [-0.5, 0.9] is good
    z = box_position[2]
    base_start_position = -box_position[:2]

    # # joint limit of panda, from https://frankaemika.github.io/docs/control_parameters.html
    # ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    # ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

    # initial joint position
    joint_start_position = np.array([0.0, -np.pi / 4, 0.0, -np.pi, 0.0, np.pi * 3 / 4, np.pi / 4,])
    joint_start_velocity = np.zeros(7)
    robot_path = "robot_data/panda_5_joint_dense_1_dataset_15"
    experiment_path = "object_data/brt_gravity_only"
    gravity = -9.81

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    urdf_path = "franka_panda/panda.urdf"
    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    # get initial guess
    q_candidates, phi_candidates, throw_candidates = brt_chunk_robot_data_matching(z, robot_path=robot_path, brt_path=experiment_path)
    q_candidates = np.array(q_candidates)
    phi_candidates = np.array(phi_candidates)
    throw_candidates = np.array(throw_candidates)

    n_candidates = q_candidates.shape[0]

    # get full throwing configuration and trajectories
    traj_durations = []
    trajs = []
    throw_configs = []
    st = time.time()
    for i in range(n_candidates):
        candidate_idx = i
        throw_config = get_throw_config(robot, q_candidates[candidate_idx],
                                                     phi_candidates[candidate_idx],
                                                     throw_candidates[candidate_idx])
        # filter out throwing configuration that will hit gripper palm
        if throw_config[4][2] < -0.02:
            continue
        # calculate throwing trajectory
        traj_throw = get_traj_from_ruckig(joint_start_position=joint_start_position, joint_start_velocity=joint_start_velocity,
                                          joint_throw_position=throw_config[0], joint_throw_velocity=throw_config[3],
                                          base_start_position=base_start_position, based =-throw_config[-1][:-1])
        
        traj_durations.append(traj_throw.duration)
        trajs.append(traj_throw)
        throw_configs.append(throw_config)

    print("Given query z=", "{0:0.2f}".format(z), ", found", len(throw_configs),
          "good throws in", "{0:0.2f}".format(1000 * (time.time() - st)), "ms")

    # select the minimum-time trajectory to simulate
    selected_idx = np.argmin(traj_durations)
    traj_throw = trajs[selected_idx]
    throw_config = throw_configs[selected_idx]

    # Other option: select the one with maximum range
    # selected_idx = np.argmin(throw_candidates[:, 0])
    # throw_config = get_throw_config(robot, q_candidates[selected_idx],
    #                                                  phi_candidates[selected_idx],
    #                                                  throw_candidates[selected_idx])
    # traj_throw = get_traj_from_ruckig(joint_start_position=joint_start_position, joint_start_velocity=joint_start_velocity, joint_throw_position=throw_config[0], joint_throw_velocity=throw_config[3],
    #                                       base_start_position=base_start_position, based =-throw_config[-1][:-1])
    p.disconnect()

    print("box_position: ", throw_config[-1])
    print("throwing range: ", "{0:0.2f}".format(-throw_candidates[selected_idx, 0]),
          "throwing height", "{0:0.2f}".format(throw_candidates[selected_idx, 1]))
    
    throw_simulation_mobile(traj_throw, throw_config, gravity) #, video_path=video_path)

def brt_chunk_robot_data_matching(z_target_to_base, robot_path, brt_path, thres=0.1):
    """

    :param z:           z_target-z_arm_base
    :param robot_path:
    :param brt_path:
    :return:
    """
    # Given target position, find out initial guesses of (q, phi, x), that is to be feed to Ruckig
    st = time.time()
    phis = np.linspace(-90, 90, 13)
    # robot_zs = np.load(robot_path + '/robot_zs.npy')
    robot_zs = np.arange(start=0.0, stop=1.10+0.01, step=0.05)
    num_robot_zs = robot_zs.shape[0]
    mesh = np.load(robot_path+'/qs.npy')
    robot_phi_gamma_velos_naive = np.load(robot_path + '/phi_gamma_velos_naive.npy')
    robot_phi_gamma_q_idxs_naive = np.load(robot_path + '/phi_gamma_q_idxs_naive.npy')
    num_gammas = robot_phi_gamma_q_idxs_naive.shape[2]

    brt_zs = np.load(brt_path + '/brt_zs.npy')
    brt_z_min = np.min(brt_zs)
    num_brt_zs = brt_zs.shape[0]
    shift_idx = round((z_target_to_base+brt_z_min) / 0.05)
    with open (brt_path + '/brt_chunk.pkl', 'rb') as fp:
        brt_chunk = pickle.load(fp)
    q_candidates = []
    phi_candidates = []
    x_candidates = []
    for i, z in enumerate(robot_zs):
        if i-shift_idx > num_brt_zs-1:
            continue
        for k in range(num_gammas):
            brt_data_z_gamma = brt_chunk[i-shift_idx][k]
            if brt_data_z_gamma is None:
                continue
            for j, phi in enumerate(phis):
                # adaptive cutoff according to max velo
                max_velo = robot_phi_gamma_velos_naive[i, j, k]
                brt_candidate = brt_data_z_gamma[brt_data_z_gamma[:, 4]<max_velo-thres]
                if brt_candidate.shape[0] > 0:
                    assert np.max(brt_candidate[:, 4]) < max_velo - thres
                    n = brt_candidate.shape[0]
                    q_add = [mesh[robot_phi_gamma_q_idxs_naive[i,j,k].astype(int), :].flatten()] * n
                    phi_add = [phi] * n
                    x_add = list(brt_candidate[:, :-1])
                    q_candidates = q_candidates + q_add
                    phi_candidates = phi_candidates + phi_add
                    x_candidates = x_candidates + x_add
    print("Given query z=", "{0:0.2f}".format(z_target_to_base) , ", found", len(q_candidates),
          "initial guesses in", "{0:0.2f}".format(1000 * (time.time() - st)), "ms")

    return  q_candidates, phi_candidates, x_candidates

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
    
    eef_velocity = np.array([velocity_direction[0] * r_dot, velocity_direction[1] * r_dot, z_dot])
    joint_velocities = J_inv @ eef_velocity
    
    # Compute the box position based on the throw
    box_position = end_effector_pos + np.array([-r_throw * velocity_direction[0], 
                                                -r_throw * velocity_direction[1], 
                                                -z_throw])

    # Adjust the final joint angle to avoid hitting the gripper
    gripper_pos, gripper_orn = p.getLinkState(robot, 11)[0:2]
    inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
    
    adjusted_eef_velocity_dir = eef_velocity / np.linalg.norm(eef_velocity)
    temp_pos = end_effector_pos + adjusted_eef_velocity_dir
    block_pos_in_gripper, _ = p.multiplyTransforms(inv_gripper_pos, inv_gripper_orn, temp_pos, [0, 0, 0, 1])
    
    velocity_angle_in_gripper = np.arctan2(block_pos_in_gripper[1], block_pos_in_gripper[0])

    if abs(velocity_angle_in_gripper) < 0.5 * np.pi:
        adjusted_angle = velocity_angle_in_gripper
    else:
        adjusted_angle = velocity_angle_in_gripper - np.sign(velocity_angle_in_gripper) * np.pi

    q[-1] = adjusted_angle

    return q, phi, throw, joint_velocities, block_pos_in_gripper, eef_velocity, end_effector_pos, box_position

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
    PANDA_BASE_HEIGHT = 0.5076438625
    box_position = throw_config[-1]
    client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40, cameraTargetPosition=[0.75, -0.75, 0])

    # NOTE: need high frequency
    freq = 1000
    dt = 1.0 / freq
    p.setGravity(0, 0, gravity)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)

    AE = throw_config[-2]
    EB = box_position - AE

    controlled_joints = [3, 4, 5, 6, 7, 8, 9]
    gripper_joints = [12, 13]
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotEndEffectorIndex = 14
    robot_id = p.loadURDF("descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf", [-box_position[0], -box_position[1], 0], useFixedBase=True)

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0.0])
    soccerball_id = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    box_id = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                       [0, 0, PANDA_BASE_HEIGHT+box_position[2]],
                       globalScaling=0.5)
    p.changeDynamics(soccerball_id, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
                     spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    p.changeDynamics(plane_id, -1, restitution=0.9)
    p.changeDynamics(robot_id, gripper_joints[0], jointUpperLimit=100)
    p.changeDynamics(robot_id, gripper_joints[1], jointUpperLimit=100)

    start_time, end_time = 0, trajectory.duration
    plan_time = end_time - start_time
    sample_t = np.arange(0, end_time, dt)
    n_steps = sample_t.shape[0]
    traj_data = np.zeros([3, n_steps, 7])
    base_traj_data = np.zeros([3, n_steps, 2])
    for i in range(n_steps):
        for j in range(3):
            tmp = trajectory.at_time(sample_t[i])[j]
            traj_data[j, i] = tmp[:7]
            base_traj_data[j, i] = tmp[-2:]

    # reset the joint
    # see https://github.com/bulletphysics/bullet3/issues/2803#issuecomment-770206176
    joint_start_position = traj_data[0, 0]
    p.resetBasePositionAndOrientation(robot_id, np.append(base_traj_data[0,0], 0.0), [0, 0, 0, 1])
    p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in joint_start_position])
    eef_state = p.getLinkState(robot_id, robotEndEffectorIndex, computeLinkVelocity=1)
    p.resetBasePositionAndOrientation(soccerball_id, eef_state[0], [0, 0, 0, 1])
    p.resetJointState(robot_id, gripper_joints[0], 0.03)
    p.resetJointState(robot_id, gripper_joints[1], 0.03)
    current_time = 0
    flag = True
    while(True):
        if flag:
            ref_full = trajectory.at_time(current_time)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in ref[0]], targetVelocities=[[joint_start_position_i] for joint_start_position_i in ref[1]])
            p.resetBasePositionAndOrientation(robot_id, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        else:
            ref_full = trajectory.at_time(plan_time)
            ref = [ref_full[i][:7] for i in range(3)]
            ref_base = [ref_full[i][-2:] for i in range(3)]
            p.resetJointStatesMultiDof(robot_id, controlled_joints, [[joint_start_position_i] for joint_start_position_i in ref[0]])
            p.resetBasePositionAndOrientation(robot_id, np.append(ref_base[0], 0.0), [0, 0, 0, 1])
        if current_time > plan_time - 1*dt:
            p.resetJointState(robot_id, gripper_joints[0], 0.05)
            p.resetJointState(robot_id, gripper_joints[1], 0.05)
        else:
            eef_state = p.getLinkState(robot_id, robotEndEffectorIndex, computeLinkVelocity=1)
            p.resetBasePositionAndOrientation(soccerball_id, eef_state[0], [0, 0, 0, 1])
            p.resetBaseVelocity(soccerball_id, linearVelocity=eef_state[-2])
        p.stepSimulation()
        current_time = current_time + dt
        if current_time > trajectory.duration:
            flag = False
        time.sleep(dt)
        if current_time > 5.0:
            break
    
    p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Overall
    box_position = np.array([1.0, 0.0, 0.5])
    print(f"box position in panda frame: {box_position}")
    
    plan_and_simulate_throw(box_position)