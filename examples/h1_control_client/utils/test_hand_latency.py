#!/usr/bin/env python3
"""
Test script to measure hand control latency.

This script tests the latency between sending hand commands and the hands
actually moving. It performs several tests:

1. Step response test: Send open->close->open commands and measure timing
2. Continuous command test: Send commands at different frequencies
3. Command queue test: Send multiple commands rapidly to check buffering

Usage:
    python test_hand_latency.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add robot_control to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_control import H1_2_ArmController

# Reset pose from h1_execution_client.py (26 DOF: arms + hands)
RESET_POSE = np.array([
    -5.86500406e-01,  4.23975229e-01, -2.18959212e-01,  3.03983688e-01,
     8.37489605e-01, -1.48091733e-01,  3.26156795e-01, -6.73690319e-01,
    -2.56984234e-01,  2.46559739e-01,  5.24306059e-01, -8.21682692e-01,
    -1.56169176e-01, -5.53857088e-01,  # 14 arm joints
    9.49755615e+02,  9.94054688e+02,  9.75871399e+02,  9.71857544e+02,
    8.82695190e+02,  9.43722473e+02,  # Left hand (6 joints)
    8.38617737e+02,  9.47012756e+02,  9.35925964e+02,  9.37267151e+02,
    1.00186505e+03,  8.59005188e+02   # Right hand (6 joints)
])


def reset_to_pose(robot, duration: float = 2.0, control_freq: int = 30):
    """
    Smoothly reset the robot to the configured reset pose.
    
    Args:
        robot: H1_2_ArmController instance
        duration: Time in seconds to complete the reset motion
        control_freq: Control frequency in Hz
    """
    print(f"\nResetting robot to initial pose (duration: {duration}s)...")
    
    # Get current position
    current_q = robot.get_current_dual_arm_q()
    
    # Interpolate smoothly to reset pose (arm joints only)
    num_steps = int(duration * control_freq)
    
    for i in range(num_steps):
        t = (i + 1) / num_steps  # 0 to 1
        # Smooth interpolation (ease in-out)
        t_smooth = t * t * (3 - 2 * t)
        
        # Interpolate arm joints (first 14)
        target_q = current_q + t_smooth * (RESET_POSE[:14] - current_q)
        
        # Extract hand positions from reset pose
        left_hand = RESET_POSE[14:20]
        right_hand = RESET_POSE[20:26]
        
        robot.ctrl_dual_arm(
            q_target=target_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=left_hand,
            right_hand_gesture=right_hand
        )
        
        time.sleep(1.0 / control_freq)
    
    print("Reset complete")
    print(f"  Arm joints: {RESET_POSE[:14]}")
    print(f"  Left hand: {RESET_POSE[14:20]}")
    print(f"  Right hand: {RESET_POSE[20:26]}")


def test_step_response(robot, trials: int = 5):
    """
    Test step response: measure time for hands to react to open/close commands.
    
    Since we don't have position feedback, we use visual observation and timing.
    
    Note: Robot should already be in the reset pose (hands at ~800-1000, mostly open).
    """
    print("\n" + "=" * 80)
    print("TEST 1: Step Response Test")
    print("=" * 80)
    print(f"This test will open and close both hands {trials} times.")
    print("Watch the hands and note any delays between command and actual movement.")
    print("Starting from reset pose (hands mostly open, values ~800-1000).")
    print()
    
    input("Press Enter to start test...")
    
    open_gesture = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])  # Fully open
    closed_gesture = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Fully closed (grasping)
    
    timings = []
    
    for i in range(trials):
        print(f"\nTrial {i+1}/{trials}")
        
        # Get current arm position
        current_q = robot.get_current_dual_arm_q()
        
        # Command: OPEN hands
        print(f"  {time.strftime('%H:%M:%S.%f')[:-3]} - Sending OPEN command...")
        t_open_cmd = time.time()
        
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=open_gesture,
            right_hand_gesture=open_gesture
        )
        
        input("    Press Enter when hands start moving (or immediately if no delay)...")
        t_open_start = time.time()
        open_delay = t_open_start - t_open_cmd
        print(f"    Measured delay to start: {open_delay*1000:.1f} ms")
        
        time.sleep(1.5)  # Wait for hands to fully open
        
        # Command: CLOSE hands
        print(f"  {time.strftime('%H:%M:%S.%f')[:-3]} - Sending CLOSE command...")
        t_close_cmd = time.time()
        
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=closed_gesture,
            right_hand_gesture=closed_gesture
        )
        
        input("    Press Enter when hands start moving (or immediately if no delay)...")
        t_close_start = time.time()
        close_delay = t_close_start - t_close_cmd
        print(f"    Measured delay to start: {close_delay*1000:.1f} ms")
        
        timings.append({
            'open_delay': open_delay,
            'close_delay': close_delay
        })
        
        time.sleep(1.5)  # Wait for hands to fully close
    
    # Print summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print("-" * 80)
    open_delays = [t['open_delay'] for t in timings]
    close_delays = [t['close_delay'] for t in timings]
    
    print(f"Open command delays (ms):  min={min(open_delays)*1000:.1f}, max={max(open_delays)*1000:.1f}, avg={np.mean(open_delays)*1000:.1f}")
    print(f"Close command delays (ms): min={min(close_delays)*1000:.1f}, max={max(close_delays)*1000:.1f}, avg={np.mean(close_delays)*1000:.1f}")


def test_command_frequency(robot, control_hz: int = 30, duration: float = 10.0):
    """
    Test sending hand commands at a specific frequency.
    
    This simulates the actual execution scenario where commands are sent at 30Hz.
    We'll gradually close the hands while monitoring timing.
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: Command Frequency Test ({control_hz}Hz)")
    print("=" * 80)
    print(f"This test will send hand commands at {control_hz}Hz for {duration}s.")
    print("The hands will gradually close and then open.")
    print("Monitor for any jitter or delays.")
    print()
    
    input("Press Enter to start test...")
    
    control_period = 1.0 / control_hz
    num_steps = int(duration * control_hz)
    
    # Get current arm position
    current_q = robot.get_current_dual_arm_q()
    
    timing_data = {
        'cmd_times': [],
        'loop_times': [],
        'sleep_times': []
    }
    
    print(f"Starting at {time.strftime('%H:%M:%S')}")
    test_start = time.time()
    
    for i in range(num_steps):
        loop_start = time.time()
        
        # Calculate hand position: 1000->0 (open to closed) in first half, 0->1000 (closed to open) in second half
        t = i / num_steps
        if t < 0.5:
            hand_value = 1000.0 - (t / 0.5) * 1000.0  # 1000 (open) to 0 (closed)
        else:
            hand_value = ((t - 0.5) / 0.5) * 1000.0  # 0 (closed) to 1000 (open)
        
        hand_gesture = np.full(6, hand_value)
        
        # Send command
        cmd_time = time.time()
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=hand_gesture,
            right_hand_gesture=hand_gesture
        )
        
        timing_data['cmd_times'].append(cmd_time - test_start)
        
        # Sleep to maintain frequency
        elapsed = time.time() - loop_start
        sleep_time = max(0, control_period - elapsed)
        timing_data['loop_times'].append(elapsed)
        timing_data['sleep_times'].append(sleep_time)
        
        time.sleep(sleep_time)
        
        # Print progress every second
        if i % control_hz == 0:
            print(f"  {i//control_hz}s: hand_value={hand_value:.0f}, loop_time={elapsed*1000:.1f}ms")
    
    test_end = time.time()
    
    # Print timing analysis
    print("\n" + "-" * 80)
    print("TIMING ANALYSIS:")
    print("-" * 80)
    print(f"Total duration: {test_end - test_start:.3f}s (target: {duration}s)")
    print(f"Average loop time: {np.mean(timing_data['loop_times'])*1000:.3f}ms")
    print(f"Max loop time: {np.max(timing_data['loop_times'])*1000:.3f}ms")
    print(f"Min loop time: {np.min(timing_data['loop_times'])*1000:.3f}ms")
    print(f"Target loop time: {control_period*1000:.3f}ms")
    
    # Check for timing violations
    violations = [t for t in timing_data['loop_times'] if t > control_period]
    if violations:
        print(f"\nWARNING: {len(violations)} timing violations (loop took longer than {control_period*1000:.1f}ms)")
        print(f"  Worst violation: {max(violations)*1000:.1f}ms")


def test_command_buffering(robot, num_commands: int = 10):
    """
    Test command buffering by sending multiple commands rapidly.
    
    This checks if the hand controller buffers commands or if they get lost.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Command Buffering Test")
    print("=" * 80)
    print(f"This test will send {num_commands} commands as fast as possible.")
    print("We'll watch if the hands execute all commands or if some get dropped.")
    print()
    
    input("Press Enter to start test...")
    
    current_q = robot.get_current_dual_arm_q()
    
    # Create a sequence of hand positions (from open to closed)
    positions = np.linspace(1000, 0, num_commands)  # 1000 (open) to 0 (closed)
    
    print("Sending commands rapidly...")
    send_start = time.time()
    send_times = []
    
    for i, pos in enumerate(positions):
        cmd_start = time.time()
        hand_gesture = np.full(6, pos)
        
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=hand_gesture,
            right_hand_gesture=hand_gesture
        )
        
        cmd_end = time.time()
        send_times.append(cmd_end - cmd_start)
        
        print(f"  Command {i+1}/{num_commands}: pos={pos:.0f}, send_time={send_times[-1]*1000:.3f}ms")
    
    send_end = time.time()
    
    print("\n" + "-" * 80)
    print("SEND TIMING:")
    print("-" * 80)
    print(f"Total send time: {send_end - send_start:.3f}s")
    print(f"Average command send time: {np.mean(send_times)*1000:.3f}ms")
    print(f"Commands per second: {num_commands/(send_end-send_start):.1f} Hz")
    
    print("\nWait 3 seconds for hands to finish moving...")
    time.sleep(3.0)
    
    print("\nOBSERVATION: Did the hands smoothly move through all positions,")
    print("            or did they jump to the final position?")


def test_hand_bridge_latency(robot):
    """
    Test the hand bridge communication latency directly.
    
    This checks the round-trip time for hand commands.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Hand Bridge Communication Latency")
    print("=" * 80)
    print("This test measures the time to send commands to the hand bridges.")
    print()
    
    if not hasattr(robot, 'left_hand_bridge') or not hasattr(robot, 'right_hand_bridge'):
        print("ERROR: Hand bridges not available")
        return
    
    input("Press Enter to start test...")
    
    # Test left hand
    print("\nTesting LEFT hand bridge...")
    left_times = []
    test_gesture = np.array([500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
    
    for i in range(10):
        start = time.time()
        robot.left_hand_bridge.set_gesture(test_gesture)
        end = time.time()
        elapsed = (end - start) * 1000
        left_times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.3f}ms")
        time.sleep(0.1)
    
    # Test right hand
    print("\nTesting RIGHT hand bridge...")
    right_times = []
    
    for i in range(10):
        start = time.time()
        robot.right_hand_bridge.set_gesture(test_gesture)
        end = time.time()
        elapsed = (end - start) * 1000
        right_times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.3f}ms")
        time.sleep(0.1)
    
    print("\n" + "-" * 80)
    print("COMMUNICATION LATENCY:")
    print("-" * 80)
    print(f"Left hand:  avg={np.mean(left_times):.3f}ms, max={np.max(left_times):.3f}ms")
    print(f"Right hand: avg={np.mean(right_times):.3f}ms, max={np.max(right_times):.3f}ms")
    
    if np.mean(left_times) > 50 or np.mean(right_times) > 50:
        print("\nWARNING: Communication latency is high (>50ms)")
        print("This could cause noticeable delays during execution.")


def main():
    parser = argparse.ArgumentParser(description="Test hand control latency")
    parser.add_argument("--left-hand-ip", type=str, default="192.168.123.211",
                       help="Left hand IP address")
    parser.add_argument("--right-hand-ip", type=str, default="192.168.123.210",
                       help="Right hand IP address")
    parser.add_argument("--network-interface", type=str, default="eno1",
                       help="Network interface")
    parser.add_argument("--test", type=str, choices=['all', 'step', 'freq', 'buffer', 'bridge'],
                       default='all', help="Which test to run")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  Hand Control Latency Test Suite")
    print("=" * 80)
    print()
    print("This script will test various aspects of hand control latency.")
    print("Please observe the robot hands carefully during the tests.")
    print()
    
    # Initialize robot with hand control
    print("Initializing robot with hand control...")
    try:
        robot = H1_2_ArmController(
            simulation_mode=False,
            hand_control=True,
            left_hand_ip=args.left_hand_ip,
            right_hand_ip=args.right_hand_ip,
            network_interface=args.network_interface
        )
        print("Robot initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize robot: {e}")
        return 1
    
    # Move to home position first
    print("\nMoving to home position...")
    robot.ctrl_dual_arm_go_home()
    time.sleep(2.0)
    
    # Reset to the initial pose from h1_execution_client.py
    reset_to_pose(robot, duration=3.0, control_freq=30)
    time.sleep(1.0)
    
    print("\n" + "=" * 80)
    print("Robot is now in the initial execution pose.")
    print("Hand values should be around 800-1000 (mostly open).")
    print("=" * 80)
    
    # Run tests
    try:
        if args.test in ['all', 'step']:
            test_step_response(robot, trials=3)
        
        if args.test in ['all', 'freq']:
            test_command_frequency(robot, control_hz=30, duration=10.0)
        
        if args.test in ['all', 'buffer']:
            test_command_buffering(robot, num_commands=10)
        
        if args.test in ['all', 'bridge']:
            test_hand_bridge_latency(robot)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Return to initial pose, then home
        print("\nReturning to initial pose...")
        reset_to_pose(robot, duration=2.0, control_freq=30)
        time.sleep(1.0)
        print("Returning to home position...")
        robot.ctrl_dual_arm_go_home()
        time.sleep(1.0)
    
    print("\n" + "=" * 80)
    print("  Test Complete")
    print("=" * 80)
    print()
    print("SUMMARY OF FINDINGS:")
    print("  - Review the timing measurements above")
    print("  - If delays are consistently >100ms, check:")
    print("    1. Network latency (ping the hand IPs)")
    print("    2. Hand bridge performance")
    print("    3. Whether commands are being buffered")
    print("  - If delays are intermittent, check:")
    print("    1. Network congestion")
    print("    2. CPU load on robot controller")
    print("    3. Hand firmware responsiveness")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

