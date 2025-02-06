import numpy as np


def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def sample_pick_and_place_pose(seed=None):
    """Sample initial pose for cube in pick-and-place task"""
    rng = np.random.RandomState(seed)
    
    # Sample cube position on table
    x_range = [-0.1, 0.3]
    y_range = [0.25, 0.75]
    z_range = [0.05, 0.5]  # Fixed height on table
    
    ranges = np.vstack([x_range, y_range, z_range])
    position = rng.uniform(ranges[:, 0], ranges[:, 1])
    
    # Fixed upright orientation (quaternion)
    orientation = np.array([1, 0, 0, 0])
    
    return np.concatenate([position, orientation])


def sample_platform_pose(seed=None):
    """Sample a random position for the target platform."""
    rng = np.random.RandomState(seed)
    # Adjust these ranges based on your workspace constraints
    x = rng.uniform(-0.1, 0.3)
    y = rng.uniform(0.25, 0.75)
    z = rng.uniform(0.05, 0.5)
    return np.array([x, y, z])
