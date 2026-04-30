import numpy as np
import pickle
import importlib


G1_12DOF_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

G1_12DOF_DEFAULT_DOF_POS = np.asarray(
    [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0],
    dtype=np.float32,
)

G1_20DOF_JOINT_NAMES = G1_12DOF_JOINT_NAMES + [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

G1_20DOF_DEFAULT_DOF_POS = np.asarray(
    [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3],
    dtype=np.float32,
)

G1_DEFAULT_JOINT_NAMES = G1_20DOF_JOINT_NAMES
G1_DEFAULT_DOF_POS = G1_20DOF_DEFAULT_DOF_POS

G1_TIENKUNG_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

G1_TIENKUNG_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

G1_TIENKUNG_RIGHT_LEG_JOINT_NAMES = [
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

G1_TIENKUNG_LEFT_LEG_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

G1_LEFT_HAND_LINK_CANDIDATES = (
    "left_wrist_roll_rubber_hand",
    "left_rubber_hand",
    "left_wrist_yaw_link",
    "left_wrist_roll_link",
    "left_elbow_link",
)

G1_RIGHT_HAND_LINK_CANDIDATES = (
    "right_wrist_roll_rubber_hand",
    "right_rubber_hand",
    "right_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_elbow_link",
)

G1_LEFT_FOOT_LINK_CANDIDATES = (
    "left_ankle_roll_link",
    "left_toe_link",
)

G1_RIGHT_FOOT_LINK_CANDIDATES = (
    "right_ankle_roll_link",
    "right_toe_link",
)

G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def resolve_dt(fps=None, dt=None):
    if dt is not None:
        return float(dt)
    if fps is None:
        raise ValueError("Either 'fps' or 'dt' must be provided")
    fps = float(fps)
    if fps <= 0.0:
        raise ValueError("'fps' must be positive")
    return 1.0 / fps


def resolve_fps(fps=None, dt=None):
    if fps is not None:
        fps = float(fps)
        if fps <= 0.0:
            raise ValueError("'fps' must be positive")
        return fps
    dt = resolve_dt(dt=dt)
    return 1.0 / dt


def _to_numpy_array(data, dtype=np.float32):
    return np.asarray(data, dtype=dtype)


def _identity_quat(num_frames):
    quat = np.zeros((num_frames, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    return quat


def _lookup_first(data, keys, default=None):
    if data is None:
        return default
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]
        if hasattr(data, key):
            return getattr(data, key)
    return default


def unwrap_motion_container(raw_data):
    current = raw_data
    for key in ("motion", "data", "trajectory", "traj"):
        nested = _lookup_first(current, (key,))
        if nested is not None:
            current = nested
            break
    return current


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Loads numpy-backed pickles across numpy internal module renames."""

    _MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
    }

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            alias = self._MODULE_ALIASES.get(module)
            if alias is None:
                raise
            imported = importlib.import_module(alias)
            return getattr(imported, name)


def load_motion_file(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as raw_data:
            return {key: raw_data[key] for key in raw_data.files}
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as file:
            try:
                raw_data = pickle.load(file)
            except ModuleNotFoundError as exc:
                if not str(exc).startswith("No module named 'numpy._core"):
                    raise
                file.seek(0)
                raw_data = _NumpyCompatUnpickler(file).load()
        return unwrap_motion_container(raw_data)
    raise ValueError(f"Unsupported motion file format for: {path}")


def _normalize_joint_names(joint_names):
    if joint_names is None:
        return None
    normalized = []
    for name in np.asarray(joint_names).tolist():
        normalized.append(str(name))
    return normalized


def _reshape_world_points(values, num_frames):
    if values is None:
        return None
    values = _to_numpy_array(values).reshape(num_frames, -1)
    if values.size == 0:
        return None
    return values


def _find_first_present_index(link_body_list, candidate_names):
    for candidate_name in candidate_names:
        if candidate_name in link_body_list:
            return link_body_list.index(candidate_name)
    return None


def _extract_world_points_from_local_bodies(
    local_body_pos,
    link_body_list,
    root_pos,
    root_quat,
    candidate_groups,
):
    indices = []
    for candidate_names in candidate_groups:
        index = _find_first_present_index(link_body_list, candidate_names)
        if index is None:
            return None
        indices.append(index)

    local_points = local_body_pos[:, indices, :]
    repeated_quat = np.repeat(root_quat[:, None, :], len(indices), axis=1)
    world_points = quat_rotate(repeated_quat, local_points) + root_pos[:, None, :]
    return world_points.reshape(local_body_pos.shape[0], -1)


def _resolve_joint_names_for_amp(joint_names, dof_values):
    resolved_joint_names = _normalize_joint_names(joint_names)
    if resolved_joint_names is not None:
        return resolved_joint_names
    resolved_joint_names = _infer_g1_joint_names(dof_values)
    if resolved_joint_names is None:
        raise ValueError(
            "Unable to infer joint names for AMP observation construction. "
            "Please provide G1 joint names in the motion data."
        )
    return resolved_joint_names


def _select_joint_block(values, joint_names, target_joint_names):
    index_map = {name: idx for idx, name in enumerate(joint_names)}
    missing = [name for name in target_joint_names if name not in index_map]
    if missing:
        raise ValueError(f"Motion data is missing required joints for AMP observations: {missing}")
    indices = [index_map[name] for name in target_joint_names]
    return values[:, indices]


def reorder_dof_by_joint_names(dof_pos, joint_names, target_joint_names):
    joint_names = _normalize_joint_names(joint_names)
    if joint_names is None:
        raise ValueError("joint_names are required to reorder dof data")
    index_map = {name: idx for idx, name in enumerate(joint_names)}
    missing = [name for name in target_joint_names if name not in index_map]
    if missing:
        raise ValueError(f"Missing joints in motion data: {missing}")
    indices = [index_map[name] for name in target_joint_names]
    return dof_pos[:, indices]


def _infer_g1_joint_names(dof_pos):
    if dof_pos.shape[1] == len(G1_29DOF_JOINT_NAMES):
        return list(G1_29DOF_JOINT_NAMES)
    if dof_pos.shape[1] == len(G1_20DOF_JOINT_NAMES):
        return list(G1_20DOF_JOINT_NAMES)
    if dof_pos.shape[1] == len(G1_12DOF_JOINT_NAMES):
        return list(G1_12DOF_JOINT_NAMES)
    return None


def canonicalize_motion_dict(raw_data, target_joint_names=None, fps=None, dt=None):
    raw_data = unwrap_motion_container(raw_data)
    dof_pos = _lookup_first(raw_data, ("dof_pos", "joint_pos", "qpos"))
    if dof_pos is None:
        raise KeyError("Motion data must contain one of: 'dof_pos', 'joint_pos', 'qpos'")
    dof_pos = _to_numpy_array(dof_pos)

    source_joint_names = _normalize_joint_names(_lookup_first(raw_data, ("joint_names", "dof_names")))
    if source_joint_names is None:
        source_joint_names = _infer_g1_joint_names(dof_pos)
    joint_names = source_joint_names
    if target_joint_names is not None:
        if joint_names is not None:
            dof_pos = reorder_dof_by_joint_names(dof_pos, joint_names, target_joint_names)
        else:
            raise ValueError(
                "target_joint_names were requested, but the motion data does not expose joint names "
                "and its dof dimension could not be inferred."
            )
        joint_names = list(target_joint_names)
    elif joint_names is None:
        joint_names = [f"joint_{idx}" for idx in range(dof_pos.shape[1])]

    root_pos = _lookup_first(raw_data, ("root_pos", "base_pos", "pelvis_pos"))
    if root_pos is None:
        root_pos = np.zeros((dof_pos.shape[0], 3), dtype=np.float32)
    else:
        root_pos = _to_numpy_array(root_pos)

    root_quat = _lookup_first(raw_data, ("root_quat", "root_rot", "base_quat", "pelvis_quat"))
    if root_quat is None:
        root_quat = _identity_quat(dof_pos.shape[0])
    else:
        root_quat = _to_numpy_array(root_quat)

    dof_vel = _lookup_first(raw_data, ("dof_vel", "joint_vel", "qvel"))
    if dof_vel is not None:
        dof_vel = _to_numpy_array(dof_vel)
        if dof_vel.size == 0:
            dof_vel = None
        elif target_joint_names is not None and source_joint_names is not None:
            dof_vel = reorder_dof_by_joint_names(dof_vel, source_joint_names, target_joint_names)

    feet_pos_world = _lookup_first(
        raw_data,
        ("feet_pos_world", "end_effector_pos_world", "feet_pos", "end_effector_pos", "ee_pos"),
    )
    feet_pos_world = _reshape_world_points(feet_pos_world, dof_pos.shape[0])
    hand_pos_world = _lookup_first(
        raw_data,
        ("hand_pos_world", "hands_pos_world"),
    )
    hand_pos_world = _reshape_world_points(hand_pos_world, dof_pos.shape[0])

    local_body_pos = _lookup_first(raw_data, ("local_body_pos",))
    link_body_list = _normalize_joint_names(_lookup_first(raw_data, ("link_body_list", "body_names", "link_names")))
    if local_body_pos is not None and link_body_list is not None:
        local_body_pos = _to_numpy_array(local_body_pos).reshape(dof_pos.shape[0], -1, 3)
        if feet_pos_world is None:
            feet_pos_world = _extract_world_points_from_local_bodies(
                local_body_pos,
                link_body_list,
                root_pos,
                root_quat,
                (G1_LEFT_FOOT_LINK_CANDIDATES, G1_RIGHT_FOOT_LINK_CANDIDATES),
            )
        if hand_pos_world is None:
            hand_pos_world = _extract_world_points_from_local_bodies(
                local_body_pos,
                link_body_list,
                root_pos,
                root_quat,
                (G1_LEFT_HAND_LINK_CANDIDATES, G1_RIGHT_HAND_LINK_CANDIDATES),
            )

    resolved_dt = resolve_dt(fps=fps or _lookup_first(raw_data, ("fps",)), dt=dt or _lookup_first(raw_data, ("dt",)))
    resolved_fps = resolve_fps(fps=fps or _lookup_first(raw_data, ("fps",)), dt=resolved_dt)

    return {
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "feet_pos_world": feet_pos_world,
        "hand_pos_world": hand_pos_world,
        "joint_names": joint_names,
        "dt": resolved_dt,
        "fps": resolved_fps,
    }


def finite_difference(values, dt):
    values = _to_numpy_array(values)
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    return vel


def quat_conjugate(quat):
    quat = _to_numpy_array(quat)
    conj = quat.copy()
    conj[..., :3] *= -1.0
    return conj


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def quat_rotate_inverse(quat, vec):
    quat = _to_numpy_array(quat)
    vec = _to_numpy_array(vec)
    vec_quat = np.concatenate((vec, np.zeros(vec.shape[:-1] + (1,), dtype=np.float32)), axis=-1)
    rotated = quat_multiply(quat_conjugate(quat), quat_multiply(vec_quat, quat))
    return rotated[..., :3]


def quat_rotate(quat, vec):
    quat = _to_numpy_array(quat)
    vec = _to_numpy_array(vec)
    vec_quat = np.concatenate((vec, np.zeros(vec.shape[:-1] + (1,), dtype=np.float32)), axis=-1)
    rotated = quat_multiply(quat_multiply(quat, vec_quat), quat_conjugate(quat))
    return rotated[..., :3]


def world_to_local_points(root_pos, root_quat, points_world):
    points_world = _to_numpy_array(points_world).reshape(root_pos.shape[0], -1, 3)
    root_pos = _to_numpy_array(root_pos)
    root_quat = _to_numpy_array(root_quat)
    relative = points_world - root_pos[:, None, :]
    repeated_quat = np.repeat(root_quat[:, None, :], points_world.shape[1], axis=1)
    local = quat_rotate_inverse(repeated_quat, relative)
    return local.reshape(root_pos.shape[0], -1)


def build_amp_observations_from_motion(
    dof_pos,
    dof_vel,
    feet_pos_world,
    hand_pos_world,
    root_pos,
    root_quat,
    joint_names=None,
    default_dof_pos=None,
    subtract_default=True,
):
    dof_pos = _to_numpy_array(dof_pos)
    dof_vel = _to_numpy_array(dof_vel)
    if feet_pos_world is None:
        raise ValueError("feet_pos_world or end_effector_pos is required to build AMP observations")
    if hand_pos_world is None:
        raise ValueError("hand_pos_world is required to build TienKung-style AMP observations")

    joint_names = _resolve_joint_names_for_amp(joint_names, dof_pos)
    right_arm_dof_pos = _select_joint_block(dof_pos, joint_names, G1_TIENKUNG_RIGHT_ARM_JOINT_NAMES)
    left_arm_dof_pos = _select_joint_block(dof_pos, joint_names, G1_TIENKUNG_LEFT_ARM_JOINT_NAMES)
    right_leg_dof_pos = _select_joint_block(dof_pos, joint_names, G1_TIENKUNG_RIGHT_LEG_JOINT_NAMES)
    left_leg_dof_pos = _select_joint_block(dof_pos, joint_names, G1_TIENKUNG_LEFT_LEG_JOINT_NAMES)
    right_arm_dof_vel = _select_joint_block(dof_vel, joint_names, G1_TIENKUNG_RIGHT_ARM_JOINT_NAMES)
    left_arm_dof_vel = _select_joint_block(dof_vel, joint_names, G1_TIENKUNG_LEFT_ARM_JOINT_NAMES)
    right_leg_dof_vel = _select_joint_block(dof_vel, joint_names, G1_TIENKUNG_RIGHT_LEG_JOINT_NAMES)
    left_leg_dof_vel = _select_joint_block(dof_vel, joint_names, G1_TIENKUNG_LEFT_LEG_JOINT_NAMES)

    hand_pos_local = world_to_local_points(root_pos, root_quat, hand_pos_world)
    feet_pos_local = world_to_local_points(root_pos, root_quat, feet_pos_world)
    return np.concatenate(
        (
            right_arm_dof_pos,
            left_arm_dof_pos,
            right_leg_dof_pos,
            left_leg_dof_pos,
            right_arm_dof_vel,
            left_arm_dof_vel,
            right_leg_dof_vel,
            left_leg_dof_vel,
            hand_pos_local[:, 0:3],
            hand_pos_local[:, 3:6],
            feet_pos_local[:, 0:3],
            feet_pos_local[:, 3:6],
        ),
        axis=-1,
    ).astype(np.float32)
