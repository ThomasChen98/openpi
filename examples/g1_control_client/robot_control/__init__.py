"""
G1 Robot Control Module

This module provides robot control interfaces for the Unitree G1 robot.
For visualization purposes, it primarily imports from xr_teleoperate.
"""

import os
import sys

# Add xr_teleoperate to path for importing robot controllers
xr_teleoperate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'xr_teleoperate')
if os.path.exists(xr_teleoperate_path):
    sys.path.insert(0, xr_teleoperate_path)

try:
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
except ImportError as e:
    print(f"Warning: Could not import from xr_teleoperate: {e}")
    print("Robot control will not be available, but visualization should still work.")
    G1_29_ArmController = None
    G1_29_ArmIK = None
    Dex3_1_Controller = None

__all__ = ['G1_29_ArmController', 'G1_29_ArmIK', 'Dex3_1_Controller']
