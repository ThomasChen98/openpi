"""
G1 Robot Arm Inverse Kinematics

Adapted from xr_teleoperate for standalone use in openpi.
Uses Casadi + Pinocchio for IK solving.
"""

import casadi
import numpy as np
import pinocchio as pin
import os
import logging
from pinocchio import casadi as cpin

from .weighted_moving_filter import WeightedMovingFilter

logger = logging.getLogger(__name__)

# Get the directory where this file is located
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'assets')


class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # Use absolute paths to assets
        urdf_path = os.path.join(_ASSETS_DIR, 'g1', 'g1_body29_hand14.urdf')
        mesh_dir = os.path.join(_ASSETS_DIR, 'g1')
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")
        
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        self.mixed_jointsToLockIDs = [
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
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint"
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                      pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                      pin.FrameType.OP_FRAME)
        )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-6
            },
            'print_time': False,
            'calc_lam_p': False
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            try:
                import meshcat.geometry as mg
                from pinocchio.visualize import MeshcatVisualizer
                self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
                self.vis.initViewer(open=True)
                self.vis.loadViewerModel("pinocchio")
                self.vis.displayFrames(True, frame_ids=[107, 108], axis_length=0.15, axis_width=5)
                self.vis.display(pin.neutral(self.reduced_robot.model))
            except ImportError:
                logger.warning("Meshcat not available, visualization disabled")
                self.Visualization = False

        logger.info("G1_29_ArmIK initialized")

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization and self.vis is not None:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization and self.vis is not None:
                self.vis.display(sol_q)

            return sol_q, sol_tauff

        except Exception as e:
            logger.error(f"IK solver error: {e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization and self.vis is not None:
                self.vis.display(sol_q)

            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def compute_gravity_compensation(self, arm_q: np.ndarray) -> np.ndarray:
        """
        Compute feedforward torques to compensate for gravity using Pinocchio RNEA.
        
        This uses the full robot dynamics model (Recursive Newton-Euler Algorithm)
        to accurately compute gravity compensation torques for any arm configuration.
        Much more accurate than simplified trigonometric models.
        
        Args:
            arm_q: Arm joint positions (14 DOF) in the order:
                   [L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow, L_wr_roll, L_wr_pitch, L_wr_yaw,
                    R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow, R_wr_roll, R_wr_pitch, R_wr_yaw]
        
        Returns:
            tauff: Feedforward torques (14 DOF) to counteract gravity
        """
        # RNEA with zero velocity and zero acceleration gives gravity compensation
        # tau = M(q)*0 + C(q,0)*0 + g(q) = g(q)
        v = np.zeros(self.reduced_robot.model.nv)
        a = np.zeros(self.reduced_robot.model.nv)
        
        tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, arm_q, v, a)
        
        return tauff
