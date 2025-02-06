import collections

import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
    SINGLE_ARM_START_POSE,
)

# Global variables for object poses - accessible from other modules
BOX_POSE = [None]  # Will hold [x, y, z, qw, qx, qy, qz]
PLATFORM_POSE = [None]  # Will hold [x, y, z]

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


class SingleArmViperXTask(base.Task):
    """Base class for single-arm ViperX robot tasks"""
    def __init__(self, random=None):
        super().__init__(random=random)
    
    def before_step(self, action, physics):
        """Converts normalized actions to actual joint commands."""
        # Split action into arm and gripper components
        arm_action = action[:6]  # 6 joint positions
        normalized_gripper_action = action[6]  # 1 normalized gripper position
        
        # Convert normalized gripper position to actual gripper position
        gripper_action = unnormalize_puppet_gripper_position(normalized_gripper_action)
        
        # Create mirrored gripper action (one value for each finger)
        full_gripper_action = [gripper_action, -gripper_action]
        
        # Combine arm and gripper actions
        env_action = np.concatenate([arm_action, full_gripper_action])
        
        # Apply the action in physics
        super().before_step(env_action, physics)

    def get_observation(self, physics):
        """Returns an observation of the state and camera images."""
        obs = {
            "qpos": self.get_qpos(physics),
            "qvel": self.get_qvel(physics),
            "env_state": self.get_env_state(physics),
        }

        cube_pos = physics.named.data.qpos[-7:-4]  # xyz position of cube
        target_pos = physics.named.data.geom_xpos[physics.model.name2id("blue_target", "geom")]
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")
        return obs

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:6]  # Just one arm's joint positions
        gripper_qpos = [normalize_puppet_gripper_position(qpos_raw[6])]
        return np.concatenate([arm_qpos, gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel = qvel_raw[:6]
        gripper_qvel = [normalize_puppet_gripper_velocity(qvel_raw[6])]
        return np.concatenate([arm_qvel, gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickAndPlaceTask(SingleArmViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            # Use SINGLE_ARM_START_POSE instead of slicing START_ARM_POSE
            #     "waist",
            #     "shoulder",
            #     "elbow",
            #     "forearm_roll",
            #     "wrist_angle",
            #     "wrist_rotate",
            #     # normalized gripper position (0: close, 1: open)
            #     "gripper_a",
            #     "gripper_b",
            physics.named.data.qpos[:8] = SINGLE_ARM_START_POSE
            np.copyto(physics.data.ctrl[:8], SINGLE_ARM_START_POSE)
            
            # Set cube position (possibly in the air)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]  # position and orientation of cube
            
            # Get cube position and place holder underneath
            cube_pos = physics.named.data.qpos[-7:-4]  # xyz position of cube
            holder_id = physics.model.name2id("holder", "body")

            cube_geom_id = physics.model.name2id("red_cube", "geom")
            holder_geom_id = physics.model.name2id("holder_platform", "geom")
            
            cube_size = physics.model.geom_size[cube_geom_id][0]  # Get first dimension (they're all equal for cube)
            holder_thickness = physics.model.geom_size[holder_geom_id][2]  # Get z-dimension
            
            # Place holder directly under cube, accounting for dimensions
            holder_id = physics.model.name2id("holder", "body")
            physics.model.body_pos[holder_id] = [
                cube_pos[0],  # same x as cube
                cube_pos[1],  # same y as cube
                cube_pos[2] - (cube_size + holder_thickness)  # Align surfaces exactly
            ]
            # Set platform position
            assert PLATFORM_POSE[0] is not None
            target_id = physics.model.name2id("target", "body")
            physics.model.body_pos[target_id] = PLATFORM_POSE[0]
        self._initial_reward = None
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """Get cube and target positions"""
        cube_pos = physics.named.data.qpos[-7:-4]  # xyz position of cube
        target_pos = physics.named.data.geom_xpos[physics.model.name2id("blue_target", "geom")]
        return np.concatenate([cube_pos, target_pos])

    def get_reward(self, physics):
        """Compute reward based on cube position and gripper contact"""
        # Get positions
        cube_pos = physics.named.data.qpos[-7:-4]
        target_pos = physics.named.data.geom_xpos[physics.model.name2id("blue_target", "geom")]
        platform_pos = physics.named.data.geom_xpos[physics.model.name2id("holder_platform", "geom")]
        
        # Get contacts
        all_contacts = []
        for i_contact in range(physics.data.ncon):
            geom1 = physics.model.id2name(physics.data.contact[i_contact].geom1, "geom")
            geom2 = physics.model.id2name(physics.data.contact[i_contact].geom2, "geom")
            all_contacts.append((geom1, geom2))

        # Check if gripper is touching cube
        touching_cube = ("red_cube", "vx300s_left/10_left_gripper_finger") in all_contacts
        
        # Check if cube is on target
        cube_above_target = (
            np.linalg.norm(cube_pos[:2] - target_pos[:2]) < 0.05  # xy-distance
            and abs(cube_pos[2] - target_pos[2]) < 0.1  # z-distance
        )

        # Reward logic
        reward = 0
        if touching_cube:
            reward = 1
        if touching_cube and cube_pos[2] - platform_pos[2] > 0.1:  # lifted
            reward = 3
        if cube_above_target:  # successfully placed
            reward = 4 # 4 means success. See env.py

        if self._initial_reward is None:
            self._initial_reward = reward
        if reward != 0:
            print(f'------------- Unleveled Reward: {reward}, Initial: {self._initial_reward}, cube: {cube_pos}, target: {target_pos}, platform: {platform_pos}, contacts: {all_contacts}')
        if reward <= self._initial_reward:
            reward = 0
        return reward
