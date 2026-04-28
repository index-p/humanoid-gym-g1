import numpy as np
import pickle


G1_DEFAULT_JOINT_NAMES = [
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

G1_DEFAULT_DOF_POS = np.asarray(
    [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0],
    dtype=np.float32,
)


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


def load_motion_file(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as raw_data:
            return {key: raw_data[key] for key in raw_data.files}
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as file:
            raw_data = pickle.load(file)
        return unwrap_motion_container(raw_data)
    raise ValueError(f"Unsupported motion file format for: {path}")


def _normalize_joint_names(joint_names):
    if joint_names is None:
        return None
    normalized = []
    for name in np.asarray(joint_names).tolist():
        normalized.append(str(name))
    return normalized


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


def canonicalize_motion_dict(raw_data, target_joint_names=None, fps=None, dt=None):
    raw_data = unwrap_motion_container(raw_data)
    dof_pos = _lookup_first(raw_data, ("dof_pos", "joint_pos", "qpos"))
    if dof_pos is None:
        raise KeyError("Motion data must contain one of: 'dof_pos', 'joint_pos', 'qpos'")
    dof_pos = _to_numpy_array(dof_pos)

    source_joint_names = _normalize_joint_names(_lookup_first(raw_data, ("joint_names", "dof_names")))
    joint_names = source_joint_names
    if target_joint_names is not None:
        if joint_names is not None:
            dof_pos = reorder_dof_by_joint_names(dof_pos, joint_names, target_joint_names)
        joint_names = list(target_joint_names)
    elif joint_names is None:
        joint_names = [f"joint_{idx}" for idx in range(dof_pos.shape[1])]

    root_pos = _lookup_first(raw_data, ("root_pos", "base_pos", "pelvis_pos"))
    if root_pos is None:
        root_pos = np.zeros((dof_pos.shape[0], 3), dtype=np.float32)
    else:
        root_pos = _to_numpy_array(root_pos)

    root_quat = _lookup_first(raw_data, ("root_quat", "base_quat", "pelvis_quat"))
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
    if feet_pos_world is not None:
        feet_pos_world = _to_numpy_array(feet_pos_world).reshape(dof_pos.shape[0], -1)
        if feet_pos_world.size == 0:
            feet_pos_world = None

    resolved_dt = resolve_dt(fps=fps or _lookup_first(raw_data, ("fps",)), dt=dt or _lookup_first(raw_data, ("dt",)))
    resolved_fps = resolve_fps(fps=fps or _lookup_first(raw_data, ("fps",)), dt=resolved_dt)

    return {
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "feet_pos_world": feet_pos_world,
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
    root_pos,
    root_quat,
    default_dof_pos=None,
    subtract_default=True,
):
    dof_pos = _to_numpy_array(dof_pos)
    dof_vel = _to_numpy_array(dof_vel)
    if feet_pos_world is None:
        raise ValueError("feet_pos_world or end_effector_pos is required to build AMP observations")

    if subtract_default:
        if default_dof_pos is None:
            default_dof_pos = G1_DEFAULT_DOF_POS
        dof_pos_features = dof_pos - _to_numpy_array(default_dof_pos)[None, :]
    else:
        dof_pos_features = dof_pos

    feet_pos_local = world_to_local_points(root_pos, root_quat, feet_pos_world)
    return np.concatenate((dof_pos_features, dof_vel, feet_pos_local), axis=-1).astype(np.float32)
