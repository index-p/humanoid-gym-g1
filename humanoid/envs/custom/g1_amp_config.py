# SPDX-License-Identifier: BSD-3-Clause

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1AMPCfg(LeggedRobotCfg):
    """20DOF G1 task using the g1_ppo reward style with AMPPPO training."""

    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 71
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 105
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 20
        num_envs = 1024
        episode_length_s = 24
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_20dof.urdf"
        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        terminate_after_contacts_on = ["pelvis", "torso", "head", "shoulder", "elbow", "wrist"]
        penalize_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.80]
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.3,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            "hip_roll": 100,
            "hip_pitch": 100,
            "hip_yaw": 100,
            "knee": 150,
            "ankle": 40,
            "shoulder_pitch": 40,
            "shoulder_roll": 40,
            "shoulder_yaw": 40,
            "elbow": 40,
        }
        damping = {
            "hip_roll": 2,
            "hip_pitch": 2,
            "hip_yaw": 2,
            "knee": 4,
            "ankle": 2,
            "shoulder_pitch": 2,
            "shoulder_roll": 2,
            "shoulder_yaw": 2,
            "elbow": 2,
        }
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**21
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand:
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
        num_commands = 4
        resampling_time = 8.0
        heading_command = True
        zero_command_prob = 0.2

        class ranges:
            lin_vel_x = [-0.5, 1.2]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.78
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.08
        cycle_time = 0.55
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 450

        class scales:
            joint_pos = 1.6
            feet_clearance = 1.0
            feet_contact_number = 1.2
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            feet_contact_forces = -0.01
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5
            low_speed = 0.4
            track_vel_hard = 0.8
            default_joint_pos = 0.5
            orientation = 1.0
            base_height = 0.2
            base_acc = 0.2
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0


class G1AMPCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "AMPOnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "AMPPPO"
        num_steps_per_env = 60
        max_iterations = 3001
        save_interval = 100
        experiment_name = "G1_amp_20dof"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class amp:
        motion_files_display = [
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Female1Walking_c3d/*.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male1Walking_c3d/*.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_visualization/Male2Walking_c3d/*.npz",
        ]
        motion_files = [
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Female1Walking_c3d/*.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male1Walking_c3d/*.npz",
            "{LEGGED_GYM_ROOT_DIR}/data/motion_amp_expert/Male2Walking_c3d/*.npz",
        ]
        amp_reward_coef = 0.5
        amp_task_reward_lerp = 0.4
        amp_discr_hidden_dims = [512, 256]
        amp_discr_learning_rate = 5e-6
        amp_discr_batch_size = 2048
        amp_replay_buffer_size = 100000
        amp_grad_penalty_coef = 10.0
        amp_num_preload_transitions = 20000
        amp_normalize_obs = True
        amp_norm_epsilon = 1e-5
