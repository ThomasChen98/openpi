import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

# Inspire Hand SDK imports
from inspire_sdkpy import inspire_hand_defaut, inspire_dds, inspire_sdk
from inspire_sdkpy.inspire_dds import inspire_hand_ctrl, inspire_hand_state

import logging
logger_mp = logging.getLogger(__name__)

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

H1_2_Num_Motors = 35

# Hand control constants
INSPIRE_HAND_OPEN = 1000      # Fully open hand
INSPIRE_HAND_CLOSED = 0       # Fully closed hand
INSPIRE_HAND_DOF_PER_HAND = 6  # 6 DOF per hand

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

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

class H1_2_ArmController:
    def __init__(self, simulation_mode = False, hand_control = False, 
                 left_hand_ip = "192.168.123.211", right_hand_ip = "192.168.123.210",
                 network_interface = "eno1"):
        self.simulation_mode = simulation_mode
        self.hand_control = hand_control
        self.left_hand_ip = left_hand_ip
        self.right_hand_ip = right_hand_ip
        self.network_interface = network_interface
        
        logger_mp.info("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        
        # Hand control parameters - default to open position
        self.left_hand_gesture = np.full(INSPIRE_HAND_DOF_PER_HAND, INSPIRE_HAND_OPEN)
        self.right_hand_gesture = np.full(INSPIRE_HAND_DOF_PER_HAND, INSPIRE_HAND_OPEN)
        self.left_hand_pub = None
        self.right_hand_pub = None
        self.left_bridge_handler = None
        self.right_bridge_handler = None
        self.bridge_threads = []
        self.bridge_running = False

        # initialize lowcmd publisher and lowstate subscriber
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_2_ArmController] Subscribe dds ok.")
        
        # Initialize hand DDS communication if hand control is enabled
        if self.hand_control:
            self.init_hand_control()

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...\n")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!\n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_2_ArmController OK!\n")
    
    def init_hand_control(self):
        """Initialize hand DDS communication and bridges"""
        try:
            logger_mp.info("Initializing hand control...")
            
            # Create hand DDS publishers
            self.left_hand_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_hand_ctrl)
            self.left_hand_pub.Init()
            
            self.right_hand_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_hand_ctrl)
            self.right_hand_pub.Init()
            
            logger_mp.info("Hand DDS communication initialized")
            
            # Initialize bridges
            self.init_hand_bridges()
            
        except Exception as e:
            logger_mp.error(f"Failed to initialize hand control: {e}")
            self.hand_control = False
    
    def init_hand_bridges(self):
        """Initialize bridges that connect DDS to physical hands"""
        try:
            logger_mp.info("Initializing hand bridges...")
            
            # Create bridge handlers
            self.left_bridge_handler = inspire_sdk.ModbusDataHandler(
                ip=self.left_hand_ip,
                LR='l',
                device_id=1,
                initDDS=False,  # We already initialized DDS
                network=self.network_interface
            )
            logger_mp.info(f"Left hand bridge created (IP: {self.left_hand_ip})")
            
            self.right_bridge_handler = inspire_sdk.ModbusDataHandler(
                ip=self.right_hand_ip,
                LR='r',
                device_id=1,
                initDDS=False,  # We already initialized DDS
                network=self.network_interface
            )
            logger_mp.info(f"Right hand bridge created (IP: {self.right_hand_ip})")
            
            # Start bridge threads
            self.start_hand_bridge_threads()
            
        except Exception as e:
            logger_mp.error(f"Failed to initialize bridges: {e}")
            logger_mp.warning("Hands will not be controlled - check hand IPs and network")
    
    def start_hand_bridge_threads(self):
        """Start bridge threads for both hands"""
        self.bridge_running = True
        
        # Left hand bridge thread
        left_thread = threading.Thread(
            target=self.run_hand_bridge,
            args=(self.left_bridge_handler, "Left Hand"),
            daemon=True
        )
        left_thread.start()
        self.bridge_threads.append(left_thread)
        
        # Right hand bridge thread
        right_thread = threading.Thread(
            target=self.run_hand_bridge,
            args=(self.right_bridge_handler, "Right Hand"),
            daemon=True
        )
        right_thread.start()
        self.bridge_threads.append(right_thread)
        
        logger_mp.info("Both hand bridges started")
        time.sleep(1)  # Let bridges initialize

    def run_hand_bridge(self, handler, name):
        """Run a single bridge (DDS ↔ Modbus)"""
        logger_mp.info(f"{name} bridge thread started")
        
        while self.bridge_running:
            try:
                # This reads from physical hand and publishes state to DDS
                # Also listens for DDS commands and sends them to physical hand
                data = handler.read()
                time.sleep(0.02)  # ~50Hz
                
            except Exception as e:
                logger_mp.error(f"{name} bridge error: {e}")
                time.sleep(1)

    def stop_hand_bridges(self):
        """Stop all bridge threads"""
        logger_mp.info("Stopping hand bridges...")
        self.bridge_running = False
        for thread in self.bridge_threads:
            thread.join(timeout=2)
        logger_mp.info("Hand bridges stopped")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
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
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            # Debug: Log what we're sending to DDS (throttled)
            if not hasattr(self, '_ctrl_iter'):
                self._ctrl_iter = 0
            self._ctrl_iter += 1
            if self._ctrl_iter % 125 == 0:  # Log every 0.5s at 250Hz
                current_state = self.get_current_dual_arm_q()
                logger_mp.info(f"DEBUG DDS publish (iter {self._ctrl_iter}):")
                logger_mp.info(f"  Commanding: L_ShoulderPitch[13]={self.msg.motor_cmd[H1_2_JointIndex.kLeftShoulderPitch].q:.4f}, R_ShoulderPitch[20]={self.msg.motor_cmd[H1_2_JointIndex.kRightShoulderPitch].q:.4f}")
                logger_mp.info(f"  Current:    L_ShoulderPitch={current_state[0]:.4f}, R_ShoulderPitch={current_state[7]:.4f}")
                logger_mp.info(f"  Gains:      L_kp={self.msg.motor_cmd[H1_2_JointIndex.kLeftShoulderPitch].kp:.1f}, L_kd={self.msg.motor_cmd[H1_2_JointIndex.kLeftShoulderPitch].kd:.1f}")

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
            
            # Send hand commands if hand control is enabled
            if self.hand_control:
                self.send_hand_commands()

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target, left_hand_gesture=None, right_hand_gesture=None):
        '''Set control target values q & tau of the left and right arm motors, and hand gestures.'''
        
        # Debug: Log what we're receiving (throttled to every 0.5s)
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        current_time = time.time()
        if current_time - self._last_log_time > 0.5:
            logger_mp.info(f"DEBUG ctrl_dual_arm received q_target:")
            logger_mp.info(f"  [0] L_ShoulderPitch: {q_target[0]:.4f}")
            logger_mp.info(f"  [1] L_ShoulderRoll:  {q_target[1]:.4f}")
            logger_mp.info(f"  [7] R_ShoulderPitch: {q_target[7]:.4f}")
            logger_mp.info(f"  [8] R_ShoulderRoll:  {q_target[8]:.4f}")
            self._last_log_time = current_time
        
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            if left_hand_gesture is not None:
                self.left_hand_gesture = np.clip(left_hand_gesture, INSPIRE_HAND_CLOSED, INSPIRE_HAND_OPEN)
            if right_hand_gesture is not None:
                self.right_hand_gesture = np.clip(right_hand_gesture, INSPIRE_HAND_CLOSED, INSPIRE_HAND_OPEN)
    
    def send_hand_commands(self):
        """Send hand commands at 250Hz - same as robot control frequency"""
        if self.left_hand_pub is None or self.right_hand_pub is None:
            return
            
        # Copy target hand angles
        left_hand_angle = self.left_hand_gesture.copy()
        right_hand_angle = self.right_hand_gesture.copy()
        
        # Create left hand command (angle mode)
        left_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd.angle_set = [int(angle) for angle in left_hand_angle]
        left_cmd.pos_set = [0] * INSPIRE_HAND_DOF_PER_HAND    # Not used in angle mode
        left_cmd.force_set = [0] * INSPIRE_HAND_DOF_PER_HAND  # No force control
        left_cmd.speed_set = [0] * INSPIRE_HAND_DOF_PER_HAND  # No speed control
        left_cmd.mode = 0b0001  # Angle mode (Mode 1)
        
        # Create right hand command (angle mode)
        right_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd.angle_set = [int(angle) for angle in right_hand_angle]
        right_cmd.pos_set = [0] * INSPIRE_HAND_DOF_PER_HAND    # Not used in angle mode
        right_cmd.force_set = [0] * INSPIRE_HAND_DOF_PER_HAND  # No force control
        right_cmd.speed_set = [0] * INSPIRE_HAND_DOF_PER_HAND  # No speed control
        right_cmd.mode = 0b0001  # Angle mode (Mode 1)
        
        # Send commands
        self.left_hand_pub.Write(left_cmd)
        self.right_hand_pub.Write(right_cmd)

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position (zombie pose - arms extended forward).'''
        logger_mp.info("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        
        # Increase velocity limit for faster homing
        self.speed_instant_max()
        
        max_attempts = 100
        current_attempts = 0
        
        # Zombie pose: arms extended straight forward
        home_q = np.zeros(14)
        home_q[0] = -1.57  # Left shoulder pitch (forward)
        home_q[7] = -1.57  # Right shoulder pitch (forward)
        
        with self.ctrl_lock:
            self.q_target = home_q
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            target_diff = np.abs(current_q - home_q)
            if np.all(target_diff < tolerance):
                logger_mp.info("[H1_2_ArmController] both arms have reached the home position (zombie pose).")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

if __name__ == "__main__":
    from robot_arm_ik import H1_2_ArmIK
    import pinocchio as pin 

    arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False, Hand_Control = True)
    arm = H1_2_ArmController(hand_control = True)

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration
    
    # Hand gesture examples (6 DOF per hand, range 0-1000)
    left_hand_gesture = np.array([500, 500, 500, 500, 500, 500])  # Half open
    right_hand_gesture = np.array([500, 500, 500, 500, 500, 500])  # Half open

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()
            
            # Vary hand gestures over time
            hand_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * step / 50)
            left_hand_gesture = np.full(6, int(hand_pattern * 1000))
            right_hand_gesture = np.full(6, int(hand_pattern * 1000))
            left_hand_gesture = np.clip(left_hand_gesture, INSPIRE_HAND_CLOSED+400, INSPIRE_HAND_OPEN)
            right_hand_gesture = np.clip(right_hand_gesture, INSPIRE_HAND_CLOSED+400, INSPIRE_HAND_OPEN)

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            result = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, 
                                   current_lr_arm_q, current_lr_arm_dq,
                                   left_hand_gesture, right_hand_gesture)
            
            if arm_ik.Hand_Control:
                sol_q, sol_tauff, left_hand_out, right_hand_out = result
                arm.ctrl_dual_arm(sol_q, sol_tauff, left_hand_out, right_hand_out)
            else:
                sol_q, sol_tauff = result
                arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)