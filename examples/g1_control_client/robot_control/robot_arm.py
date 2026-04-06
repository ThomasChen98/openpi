"""
G1 Robot Arm Controller

Adapted from xr_teleoperate for standalone use in openpi.
Controls the G1_29 robot arm via Unitree DDS.
"""

import numpy as np
import threading
import time
import logging
from enum import IntEnum
from tqdm import tqdm

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

logger = logging.getLogger(__name__)

kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None


class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class G1_29_ArmController:
    def __init__(self, motion_mode=False, simulation_mode=False, dds_already_initialized=False):
        logger.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        # Physics-based PD gains derived from motor armature (BeyondMimic approach)
        # Formula: kp = armature * (2*pi*freq)^2, kd = 2*damping_ratio*armature*(2*pi*freq)
        # Using 10Hz natural frequency and 2.0 damping ratio for energy-efficient control
        # Lower gains reduce heat generation; feedforward torques handle gravity compensation
        NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz in rad/s (~62.83)
        DAMPING_RATIO = 2.0
        
        # Motor armature values from G1 specs
        ARMATURE_5020 = 0.003609725  # Shoulder/elbow/wrist_roll motors
        ARMATURE_4010 = 0.00425      # Wrist pitch/yaw motors
        ARMATURE_7520_14 = 0.010177520  # Waist yaw motor
        
        # 5020 motors: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
        self.kp_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # ~14.25
        self.kd_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # ~0.91
        
        # Shoulder gains (pitch/roll/yaw) - increased for better stiffness
        self.kp_shoulder = self.kp_5020 * 3.0   # ~14.25 * 3 = 42.75
        self.kd_shoulder = self.kd_5020 * 2.0   # ~0.91 * 2 = 1.82
        
        # Elbow gains - increased for better stiffness
        self.kp_elbow = self.kp_5020 * 10.0     # ~14.25 * 10 = 142.5
        self.kd_elbow = self.kd_5020 * 2.0       # ~0.91 * 2 = 1.82
        
        # 4010 motors: wrist_pitch, wrist_yaw (smaller motors)
        self.kp_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # ~16.78
        self.kd_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # ~1.07
        
        # 7520 motors: waist_yaw (larger motor)
        self.kp_7520 = ARMATURE_7520_14 * NATURAL_FREQ**2  # ~40.2
        self.kd_7520 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # ~2.56
        
        # Backward compatibility aliases (used in gain assignment logic)
        self.kp_high = self.kp_7520   # For non-arm strong motors
        self.kd_high = self.kd_7520
        self.kp_low = self.kp_5020    # For arm motors (shoulder/elbow)
        self.kd_low = self.kd_5020
        self.kp_wrist = self.kp_4010  # For wrist pitch/yaw (use 5020 for wrist_roll)
        self.kd_wrist = self.kd_4010
        
        # Waist yaw control gains (uses 7520 motor)
        self.kp_waist = self.kp_7520
        self.kd_waist = self.kd_7520
        
        # Waist pitch and roll gains - higher stiffness to prevent bowing forward
        self.kp_waist_pitch_roll = 350.0
        self.kd_waist_pitch_roll = 5.0
        self.waist_yaw_target = 0.0
        self.waist_yaw_limits = [-2.618, 2.618]  # From URDF: approx +/- 150 degrees
        
        # Torque limiting for safety (prevents sustained high torque that causes overheating)
        # Based on BeyondMimic effort limits: shoulder/elbow ~25Nm, wrist ~5-25Nm
        self.torque_limit_enabled = True
        self.torque_limits = np.array([
            25.0, 25.0, 25.0, 25.0,  # L: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            25.0, 5.0, 5.0,          # L: wrist_roll, wrist_pitch, wrist_yaw
            25.0, 25.0, 25.0, 25.0,  # R: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            25.0, 5.0, 5.0,          # R: wrist_roll, wrist_pitch, wrist_yaw
        ], dtype=np.float32)

        self.all_motor_q = None
        self.arm_velocity_limit = 40.0  # Was 20.0 - doubled to allow faster tracking
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        # initialize lowcmd publisher and lowstate subscriber
        if not dds_already_initialized:
            if self.simulation_mode:
                ChannelFactoryInitialize(1)
            else:
                ChannelFactoryInitialize(0)
        else:
            logger.info("[G1_29_ArmController] DDS already initialized, skipping ChannelFactoryInitialize")

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        with tqdm(desc="[G1_29] Waiting for DDS", unit="attempt", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            while not self.lowstate_buffer.GetData():
                time.sleep(0.1)
                pbar.update(1)
        logger.info("[G1_29_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger.info(f"Current all body motor state q:\n{self.all_motor_q}")
        logger.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}")
        logger.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id == G1_29_JointIndex.kWaistYaw:
                # Waist yaw with 7520 motor gains
                self.msg.motor_cmd[id].kp = self.kp_waist
                self.msg.motor_cmd[id].kd = self.kd_waist
                self.waist_yaw_target = self.all_motor_q[id]
            elif id == G1_29_JointIndex.kWaistPitch or id == G1_29_JointIndex.kWaistRoll:
                # Waist pitch and roll with higher stiffness to prevent bowing forward
                self.msg.motor_cmd[id].kp = self.kp_waist_pitch_roll
                self.msg.motor_cmd[id].kd = self.kd_waist_pitch_roll
            elif id.value in arm_indices:
                # Arm motors use physics-based gains by motor type
                if self._Is_shoulder_motor(id):
                    # Shoulder motors (pitch/roll/yaw) with increased gains
                    self.msg.motor_cmd[id].kp = self.kp_shoulder
                    self.msg.motor_cmd[id].kd = self.kd_shoulder
                elif self._Is_elbow_motor(id):
                    # Elbow motors with increased gains
                    self.msg.motor_cmd[id].kp = self.kp_elbow
                    self.msg.motor_cmd[id].kd = self.kd_elbow
                elif self._Is_wrist_pitch_yaw_motor(id):
                    # Wrist pitch/yaw use 4010 motors
                    self.msg.motor_cmd[id].kp = self.kp_4010
                    self.msg.motor_cmd[id].kd = self.kd_4010
                else:
                    # Wrist roll uses 5020 motors
                    self.msg.motor_cmd[id].kp = self.kp_5020
                    self.msg.motor_cmd[id].kd = self.kd_5020
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        logger.info("Lock OK!")
        logger.info(f"[G1_29_ArmController] Initial waist yaw: {self.waist_yaw_target:.3f} rad")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target
                waist_yaw_target = self.waist_yaw_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]
            
            # Command waist yaw
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].q = waist_yaw_target
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].dq = 0
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].kp = self.kp_waist
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].kd = self.kd_waist
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].tau = 0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 40.0 + (10.0 * min(1.0, t_elapsed / 5.0))  # Base 40 + gradual

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target, use_gravity_compensation: bool = False):
        '''Set control target values q & tau of the left and right arm motors.
        
        Args:
            q_target: Target joint positions (14 DOF)
            tauff_target: Feedforward torques (14 DOF) - added to gravity comp if enabled
            use_gravity_compensation: If True, add simplified gravity compensation torques
                                      (prefer RNEA-based torques from IK solver instead)
        '''
        with self.ctrl_lock:
            self.q_target = q_target
            if use_gravity_compensation:
                gravity_torques = self.compute_gravity_compensation(q_target)
                final_tauff = tauff_target + gravity_torques
            else:
                final_tauff = tauff_target
            
            # Apply torque limiting for safety (prevents overheating)
            if self.torque_limit_enabled:
                final_tauff = np.clip(final_tauff, -self.torque_limits, self.torque_limits)
            
            self.tauff_target = final_tauff
    
    def compute_gravity_compensation(self, arm_q: np.ndarray) -> np.ndarray:
        """
        Compute feedforward torques to compensate for gravity on arm joints.
        
        This uses a simplified model based on joint angles. The key insight is that
        gravity torque on a joint depends on the sine of the angle between the link
        and vertical (gravity direction).
        
        Joint mapping (14 DOF):
            [0]  L_sh_pitch  - main gravity load when arm extended forward
            [1]  L_sh_roll   - gravity load when arm abducted
            [2]  L_sh_yaw    - minimal gravity load (rotation around arm axis)
            [3]  L_elbow     - gravity load on forearm
            [4-6] L_wrist    - minimal gravity load
            [7-13] Right arm (same pattern)
        
        Args:
            arm_q: Current or target arm joint positions (14 DOF)
            
        Returns:
            tauff: Feedforward torques (14 DOF) to counteract gravity
        """
        tauff = np.zeros(14, dtype=np.float32)
        
        # Empirical torque constants (Nm) - these need tuning on the real robot
        # Values represent approximate torque needed at 90° from vertical
        # Start conservative and increase if arms still drop
        SHOULDER_PITCH_TORQUE = 8.0   # Full upper arm + forearm + hand (~3kg at 0.3m)
        SHOULDER_ROLL_TORQUE = 3.0    # Lateral load (less than pitch)
        ELBOW_TORQUE = 3.0            # Forearm + hand (~1.5kg at 0.25m)
        
        # Left arm gravity compensation
        L_sh_pitch = arm_q[0]   # Positive = forward/down
        L_sh_roll = arm_q[1]    # Positive = outward
        L_elbow = arm_q[3]      # Positive = flexion
        
        # Shoulder pitch: torque needed depends on how far forward the arm is
        # sin(pitch) gives the moment arm relative to gravity
        tauff[0] = SHOULDER_PITCH_TORQUE * np.sin(L_sh_pitch)
        
        # Shoulder roll: when arm is abducted, gravity pulls it down
        # This is more complex due to interaction with pitch, simplified here
        tauff[1] = SHOULDER_ROLL_TORQUE * np.sin(L_sh_roll) * np.cos(L_sh_pitch)
        
        # Elbow: torque depends on the total angle of forearm from vertical
        # Simplified: consider forearm angle relative to upper arm direction
        forearm_angle_L = L_sh_pitch + L_elbow
        tauff[3] = ELBOW_TORQUE * np.sin(forearm_angle_L)
        
        # Right arm gravity compensation (same logic)
        R_sh_pitch = arm_q[7]
        R_sh_roll = arm_q[8]
        R_elbow = arm_q[10]
        
        tauff[7] = SHOULDER_PITCH_TORQUE * np.sin(R_sh_pitch)
        tauff[8] = SHOULDER_ROLL_TORQUE * np.sin(R_sh_roll) * np.cos(R_sh_pitch)
        
        forearm_angle_R = R_sh_pitch + R_elbow
        tauff[10] = ELBOW_TORQUE * np.sin(forearm_angle_R)
        
        return tauff

    def ctrl_waist_yaw(self, yaw_target: float):
        '''Set control target value q of the waist yaw motor.
        
        Args:
            yaw_target: Target waist yaw angle in radians (clamped to limits)
        '''
        with self.ctrl_lock:
            self.waist_yaw_target = np.clip(yaw_target, self.waist_yaw_limits[0], self.waist_yaw_limits[1])

    def get_current_waist_yaw(self) -> float:
        '''Return current state q of the waist yaw motor.'''
        return self.lowstate_buffer.GetData().motor_state[G1_29_JointIndex.kWaistYaw].q

    def get_waist_yaw_target(self) -> float:
        '''Return the current target q of the waist yaw motor.'''
        with self.ctrl_lock:
            return self.waist_yaw_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])

    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])

    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])

    def get_lowstate_raw(self):
        '''Return the raw lowstate message (includes IMU, wireless_remote, mode_machine).
        Returns None if no message has been received yet.'''
        return self.lowstate_subscriber.Read()

    def ctrl_dual_arm_go_home(self, release_control: bool = False):
        '''Move both the left and right arms and waist yaw of the robot to their home position.
        
        Args:
            release_control: If True, release SDK control to internal controller after reaching home.
                           If False (default), maintain SDK control for continued arm commands.
        '''
        logger.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            self.waist_yaw_target = 0.0  # Also reset waist yaw to zero
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            current_waist = self.get_current_waist_yaw()
            if np.all(np.abs(current_q) < tolerance) and abs(current_waist) < tolerance:
                if self.motion_mode and release_control:
                    # Only ramp down if explicitly releasing control
                    logger.info("[G1_29_ArmController] Releasing SDK arm control to internal controller...")
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger.info("[G1_29_ArmController] both arms and waist have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        '''Gradually increase velocity to maximum over t seconds.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''Set arms velocity to maximum immediately.'''
        self.arm_velocity_limit = 50.0  # Increased max velocity

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        """Check if motor is any wrist motor (roll, pitch, or yaw)."""
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

    def _Is_wrist_pitch_yaw_motor(self, motor_index):
        """Check if motor is wrist pitch or yaw (4010 motors, smaller than 5020)."""
        wrist_pitch_yaw_motors = [
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_pitch_yaw_motors

    def _Is_shoulder_motor(self, motor_index):
        """Check if motor is a shoulder motor (pitch, roll, or yaw)."""
        shoulder_motors = [
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
        ]
        return motor_index.value in shoulder_motors

    def _Is_elbow_motor(self, motor_index):
        """Check if motor is an elbow motor."""
        elbow_motors = [
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in elbow_motors


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34
