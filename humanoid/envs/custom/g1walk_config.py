# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1walkCfg(LeggedRobotCfg):
    """
    Configuration class for the G1 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 75
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 89
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 20
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_20dof.urdf'

        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"

        #terminate_after_contacts_on = ['hip','pelvis']
        terminate_after_contacts_on = ['pelvis', 'torso', 'head', 'shoulder', 'elbow', 'wrist']
        penalize_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.80]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 'left_leg_roll_joint': 0.,
            # 'left_leg_yaw_joint': 0.,
            # 'left_leg_pitch_joint': 0.,
            # 'left_knee_joint': 0.,
            # 'left_ankle_pitch_joint': 0.,
            # 'left_ankle_roll_joint': 0.,
            # 'right_leg_roll_joint': 0.,
            # 'right_leg_yaw_joint': 0.,
            # 'right_leg_pitch_joint': 0.,
            # 'right_knee_joint': 0.,
            # 'right_ankle_pitch_joint': 0.,
            # 'right_ankle_roll_joint': 0.,
            'left_hip_yaw_joint' : 0. ,   
            'left_hip_roll_joint' : 0,               
            'left_hip_pitch_joint' : -0.1,         
            'left_knee_joint' : 0.3,       
            'left_ankle_pitch_joint' : -0.2,     
            'left_ankle_roll_joint' : 0,     
            'right_hip_yaw_joint' : 0., 
            'right_hip_roll_joint' : 0, 
            'right_hip_pitch_joint' : -0.1,                                       
            'right_knee_joint' : 0.3,                                             
            'right_ankle_pitch_joint': -0.2,                              
            'right_ankle_roll_joint' : 0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.3,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.3,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'leg_roll': 200.0, 'leg_pitch': 350.0, 'leg_yaw': 200.0,
        #              'knee': 350.0, 'ankle': 15}
        # damping = {'leg_roll': 10, 'leg_pitch': 10, 'leg_yaw':
        #            10, 'knee': 10, 'ankle': 10}
        stiffness = {'hip_roll': 100,
                     'hip_pitch': 100,
                     'hip_yaw': 100,
                     'knee': 150,
                     'ankle': 40,
                     'shoulder_pitch': 40,
                     'shoulder_roll': 40,
                     'shoulder_yaw': 40,
                     'elbow': 40,
                     }  # [N*m/rad]
        damping = {  'hip_roll': 2,
                     'hip_pitch': 2,
                     'hip_yaw': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 2,
                     'elbow': 2,
                     }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        # randomize_friction = True
        # friction_range = [0.4, 2.0]
        # randomize_base_mass = True
        # added_mass_range = [-1.5, 1.5]
        # push_robots = True
        # push_interval_s = 4
        # max_push_vel_xy = 0.2
        # max_push_ang_vel = 0.4
        # # dynamic randomization
        # action_delay = 0.5
        # action_noise = 0.02
        # class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 1.0]

        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]

        push_robots = False
        push_interval_s = 6
        max_push_vel_xy = 0.05
        max_push_ang_vel = 0.05

        action_delay = 0.0
        action_noise = 0.005

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            #lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_x = [-0.5, 1.2]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            #ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            ang_vel_yaw = [-0.5, 0.5] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.78
        min_dist = 0.2
        max_dist = 0.5
        #target_feet_height = 0.06       # m
        target_feet_height = 0.08       # m
        gait_cycle = 0.55               # sec
        gait_swing_ratio = 0.38
        gait_phase_offset_l = 0.0
        gait_phase_offset_r = 0.5
        gait_transition_ratio = 0.05
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 450  # Forces above this value are penalized

        class scales:
            feet_clearance = 0.0
            feet_contact_number = 0.0
            # gait
            feet_air_time = 1.
            gait_feet_force_periodic = 1.0
            gait_feet_speed_periodic = 1.0
            gait_feet_support_periodic = 0.6
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.015
            # vel tracking
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            base_vertical_vel = -0.5
            low_speed = 0.1
            track_vel_hard = 0.2
            # base pos
            default_joint_pos = 0.0
            joint_deviation_l1 = -0.05
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.003
            hip_roll_action = -0.0005
            hip_yaw_action = -0.0001
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class G1walkCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 2e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 90  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'G1walk_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


class G1walkCfgAMPPPO(G1walkCfgPPO):
    runner_class_name = 'AMPOnPolicyRunner'

    class runner(G1walkCfgPPO.runner):
        algorithm_class_name = 'AMPPPO'
        experiment_name = 'G1walk_amp'

    class amp:
        # Keep AMP expert data focused on walking. Avoid mixing hop/leap/box/crouch/side-step
        # clips here unless training a conditioned multi-skill policy.
        motion_files_display = [
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B1_-_stand_to_walk_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B2_-_walk_to_stand_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B2_-_walk_to_stand_t2_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B3_-_walk1_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B4_-_stand_to_walk_back_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B5_-_walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B6_-_walk_backwards_to_stand_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B7_-_walk_backwards_turn_forwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B9_-_walk_turn_left_(90)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B11_-_walk_turn_left_(135)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B12_-_walk_turn_right_(90)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B13_-_walk_turn_right_(45)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B14_-_walk_turn_right_(135)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B15_-_walk_turn_around_(same_direction)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/B16_-_walk_turn_change_direction_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/Walk_B13_-_Walk_turn_right_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/Walk_B16_-_Walk_turn_change_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B10_-__Walk_turn_left_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B11_-__Walk_turn_left_135_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B13_-__Walk_turn_right_90_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B14_-__Walk_turn_right_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B14_-__Walk_turn_right_45_t2_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B15_-__Walk_turn_around_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B4_-_Stand_to_Walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B5_-__Walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/B9_-__Walk_turn_left_90_stageii.npz",
        ]
        motion_files = [
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B1_-_stand_to_walk_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B2_-_walk_to_stand_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B2_-_walk_to_stand_t2_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B3_-_walk1_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B4_-_stand_to_walk_back_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B5_-_walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B6_-_walk_backwards_to_stand_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B7_-_walk_backwards_turn_forwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B9_-_walk_turn_left_(90)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B11_-_walk_turn_left_(135)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B12_-_walk_turn_right_(90)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B13_-_walk_turn_right_(45)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B14_-_walk_turn_right_(135)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B15_-_walk_turn_around_(same_direction)_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/B16_-_walk_turn_change_direction_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/Walk_B13_-_Walk_turn_right_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/Walk_B16_-_Walk_turn_change_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B10_-__Walk_turn_left_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B11_-__Walk_turn_left_135_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B13_-__Walk_turn_right_90_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B14_-__Walk_turn_right_45_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B14_-__Walk_turn_right_45_t2_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B15_-__Walk_turn_around_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B4_-_Stand_to_Walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B5_-__Walk_backwards_stageii.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/B9_-__Walk_turn_left_90_stageii.npz",
        ]
        amp_reward_coef = 0.5
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [512, 256]
        amp_discr_learning_rate = 1e-5
        amp_discr_batch_size = 4096
        amp_replay_buffer_size = 300000
        amp_grad_penalty_coef = 10.0
        amp_num_preload_transitions = 50000
        amp_normalize_obs = True
        amp_norm_epsilon = 1e-5
