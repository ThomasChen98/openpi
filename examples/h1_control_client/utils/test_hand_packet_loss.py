#!/usr/bin/env python3
"""
Test script to validate packet loss during large hand movements.

This script monitors network connectivity to the left hand controller while
sending large movement commands to detect if hand movements cause packet loss.

Usage:
    python test_hand_packet_loss.py [--duration SECONDS] [--cycles NUMBER]
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import deque
from datetime import datetime

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
    1.00186505e+03,  8.59005188e+02   # Right hand (6 joints) - broken, will ignore
])


class PingMonitor:
    """Monitor ping responses and detect packet loss."""
    
    def __init__(self, ip_address, interval=0.01):
        self.ip_address = ip_address
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Statistics
        self.ping_times = deque(maxlen=1000)  # Keep last 1000 pings
        self.packet_loss_events = []
        self.total_pings = 0
        self.lost_pings = 0
        self.last_ping_time = None
        
        # For detecting pauses
        self.pause_threshold = 0.1  # 100ms pause = likely packet loss
        
    def start(self):
        """Start ping monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.thread.start()
        print(f"Started ping monitoring to {self.ip_address} (interval: {self.interval}s)")
        
    def stop(self):
        """Stop ping monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"Stopped ping monitoring")
        
    def _ping_loop(self):
        """Background thread that continuously pings and monitors responses."""
        # Use subprocess with continuous ping
        cmd = ['ping', '-i', str(self.interval), self.ip_address]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            prev_time = time.time()
            
            for line in iter(process.stdout.readline, ''):
                if not self.running:
                    process.terminate()
                    break
                    
                current_time = time.time()
                
                # Parse ping output
                if 'time=' in line:
                    # Successful ping
                    try:
                        time_str = line.split('time=')[1].split()[0]
                        ping_ms = float(time_str)
                        
                        self.ping_times.append({
                            'time': current_time,
                            'ping_ms': ping_ms,
                            'success': True
                        })
                        
                        # Check for pause (delay between pings)
                        if self.last_ping_time:
                            gap = current_time - self.last_ping_time
                            if gap > self.pause_threshold:
                                self.packet_loss_events.append({
                                    'time': current_time,
                                    'gap_ms': gap * 1000,
                                    'type': 'pause'
                                })
                        
                        self.last_ping_time = current_time
                        self.total_pings += 1
                        
                    except (IndexError, ValueError):
                        pass
                        
                elif 'Request timeout' in line or 'no answer' in line:
                    # Packet loss
                    self.lost_pings += 1
                    self.total_pings += 1
                    self.packet_loss_events.append({
                        'time': current_time,
                        'type': 'loss'
                    })
                    
                prev_time = current_time
                
        except Exception as e:
            print(f"Ping monitoring error: {e}")
            
    def get_stats(self):
        """Get current statistics."""
        if self.total_pings == 0:
            loss_rate = 0
        else:
            loss_rate = (self.lost_pings / self.total_pings) * 100
            
        return {
            'total_pings': self.total_pings,
            'lost_pings': self.lost_pings,
            'loss_rate': loss_rate,
            'pause_events': len([e for e in self.packet_loss_events if e['type'] == 'pause']),
            'loss_events': len([e for e in self.packet_loss_events if e['type'] == 'loss'])
        }
        
    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"\nPing Statistics:")
        print(f"  Total pings: {stats['total_pings']}")
        print(f"  Lost pings: {stats['lost_pings']}")
        print(f"  Loss rate: {stats['loss_rate']:.2f}%")
        print(f"  Pause events (>100ms gap): {stats['pause_events']}")
        print(f"  Timeout events: {stats['loss_events']}")


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
        
        # Extract left hand position from reset pose (right hand broken - use 1000)
        left_hand = RESET_POSE[14:20]
        right_hand = np.full(6, 1000.0)  # Right hand broken - always fully open
        
        robot.ctrl_dual_arm(
            q_target=target_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=left_hand,
            right_hand_gesture=right_hand
        )
        
        time.sleep(1.0 / control_freq)
    
    print("Reset complete")
    print(f"  Arm joints: {RESET_POSE[:14]}")
    print(f"  Left hand (active): {RESET_POSE[14:20]}")
    print(f"  Right hand (broken, disabled): [1000, 1000, 1000, 1000, 1000, 1000]")


def test_packet_loss_during_movement(robot, ping_monitor, cycles=5, hold_time=2.0):
    """
    Test if large hand movements cause packet loss.
    
    Args:
        robot: H1_2_ArmController instance
        ping_monitor: PingMonitor instance
        cycles: Number of open/close cycles
        hold_time: Time to hold each position (seconds)
    """
    print("\n" + "=" * 80)
    print("PACKET LOSS VALIDATION TEST")
    print("=" * 80)
    print(f"This test will perform {cycles} cycles of large hand movements")
    print("while monitoring network connectivity to the left hand controller.")
    print()
    print("Watch the ping output for:")
    print("  - Pauses in the rolling ping printout")
    print("  - Increased ping times")
    print("  - Timeout messages")
    print()
    
    input("Press Enter to start test...")
    
    # Get current arm position (keep arms still)
    current_q = robot.get_current_dual_arm_q()
    
    # Define hand gestures
    fully_open = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    fully_closed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    right_hand_disabled = np.full(6, 1000.0)  # Right hand broken - always fully open
    
    results = []
    
    for cycle in range(cycles):
        print(f"\n{'=' * 80}")
        print(f"Cycle {cycle + 1}/{cycles}")
        print(f"{'=' * 80}")
        
        # Get stats before movement
        stats_before = ping_monitor.get_stats()
        
        # OPEN command (large movement from reset position)
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Commanding LEFT hand to OPEN...")
        print("  >>> WATCH PING OUTPUT FOR PAUSES <<<")
        
        move_start = time.time()
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=fully_open,
            right_hand_gesture=right_hand_disabled
        )
        
        # Hold position and monitor
        print(f"  Holding OPEN for {hold_time}s...")
        time.sleep(hold_time)
        
        stats_after_open = ping_monitor.get_stats()
        
        # CLOSE command (large movement)
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Commanding LEFT hand to CLOSE...")
        print("  >>> WATCH PING OUTPUT FOR PAUSES <<<")
        
        robot.ctrl_dual_arm(
            q_target=current_q,
            tauff_target=np.zeros(14),
            left_hand_gesture=fully_closed,
            right_hand_gesture=right_hand_disabled
        )
        
        # Hold position and monitor
        print(f"  Holding CLOSED for {hold_time}s...")
        time.sleep(hold_time)
        
        stats_after_close = ping_monitor.get_stats()
        
        # Calculate changes during this cycle
        result = {
            'cycle': cycle + 1,
            'pings_during_open': stats_after_open['total_pings'] - stats_before['total_pings'],
            'losses_during_open': stats_after_open['lost_pings'] - stats_before['lost_pings'],
            'pauses_during_open': stats_after_open['pause_events'] - stats_before['pause_events'],
            'pings_during_close': stats_after_close['total_pings'] - stats_after_open['total_pings'],
            'losses_during_close': stats_after_close['lost_pings'] - stats_after_open['lost_pings'],
            'pauses_during_close': stats_after_close['pause_events'] - stats_after_open['pause_events'],
        }
        results.append(result)
        
        # Print immediate feedback
        print(f"\n  Cycle {cycle + 1} results:")
        print(f"    OPEN movement:  {result['losses_during_open']} losses, {result['pauses_during_open']} pauses")
        print(f"    CLOSE movement: {result['losses_during_close']} losses, {result['pauses_during_close']} pauses")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_losses = sum(r['losses_during_open'] + r['losses_during_close'] for r in results)
    total_pauses = sum(r['pauses_during_open'] + r['pauses_during_close'] for r in results)
    
    print(f"\nTotal packet losses during movements: {total_losses}")
    print(f"Total pause events (>100ms gap): {total_pauses}")
    print()
    
    for i, result in enumerate(results):
        print(f"Cycle {i+1}:")
        print(f"  OPEN:  {result['losses_during_open']} losses, {result['pauses_during_open']} pauses")
        print(f"  CLOSE: {result['losses_during_close']} losses, {result['pauses_during_close']} pauses")
    
    # Overall ping stats
    ping_monitor.print_stats()
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if total_pauses > 0 or total_losses > 0:
        print("✗ PHENOMENON VALIDATED:")
        print("  Large hand movements DO cause network disruption!")
        print(f"  - {total_pauses} pause events detected")
        print(f"  - {total_losses} packet losses detected")
        print()
        print("  This confirms that hand motor activity interferes with")
        print("  network communication to the hand controller.")
        print()
        print("  Possible causes:")
        print("    1. Electrical interference from hand motors")
        print("    2. Hand controller CPU overload during movement")
        print("    3. Shared power/ground causing voltage drops")
        print("    4. Network cable routing near motor cables")
    else:
        print("✓ No significant packet loss detected during hand movements")
        print("  The network connection remained stable throughout the test.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate packet loss during large hand movements"
    )
    parser.add_argument("--left-hand-ip", type=str, default="192.168.123.211",
                       help="Left hand IP address (default: 192.168.123.211)")
    parser.add_argument("--network-interface", type=str, default="eno1",
                       help="Network interface (default: eno1)")
    parser.add_argument("--cycles", type=int, default=5,
                       help="Number of open/close cycles (default: 5)")
    parser.add_argument("--hold-time", type=float, default=2.0,
                       help="Time to hold each position in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  Hand Movement Packet Loss Validation")
    print("=" * 80)
    print()
    print("This test validates if large hand movements cause packet loss")
    print(f"to the left hand controller at {args.left_hand_ip}")
    print()
    print("The test will:")
    print("  1. Start continuous ping monitoring")
    print("  2. Reset robot to execution pose")
    print("  3. Perform repeated large hand movements (open/close)")
    print("  4. Monitor for network disruptions during movements")
    print()
    print("NOTE: Right hand is DISABLED (broken) - will not be initialized or controlled")
    print()
    
    # Initialize ping monitor
    ping_monitor = PingMonitor(args.left_hand_ip, interval=0.01)
    ping_monitor.start()
    time.sleep(2.0)  # Let ping monitoring stabilize
    
    # Initialize robot
    print("\nInitializing robot with left hand control (right hand disabled)...")
    try:
        robot = H1_2_ArmController(
            simulation_mode=False,
            hand_control=True,
            left_hand_ip=args.left_hand_ip,
            right_hand_ip=None,  # Right hand broken - disabled
            network_interface=args.network_interface
        )
        print("✓ Robot initialized successfully (right hand disabled)")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize robot: {e}")
        ping_monitor.stop()
        return 1
    
    try:
        # Move to home position first
        print("\nMoving to home position...")
        robot.ctrl_dual_arm_go_home()
        time.sleep(2.0)
        
        # Reset to execution pose
        reset_to_pose(robot, duration=3.0, control_freq=30)
        time.sleep(1.0)
        
        print("\n✓ Robot is in initial execution pose")
        print("✓ Ping monitoring is active")
        print()
        
        # Run the test
        test_packet_loss_during_movement(
            robot, 
            ping_monitor, 
            cycles=args.cycles,
            hold_time=args.hold_time
        )
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop ping monitoring
        ping_monitor.stop()
        
        # Return to reset pose
        print("\nReturning to reset pose...")
        try:
            reset_to_pose(robot, duration=2.0, control_freq=30)
            time.sleep(1.0)
        except:
            pass
        
        print("Returning to home position...")
        try:
            robot.ctrl_dual_arm_go_home()
            time.sleep(1.0)
        except:
            pass
        
        # Cleanup
        print("Cleaning up DDS resources...")
        robot.cleanup()
        print("✓ Cleanup complete")
    
    print("\n" + "=" * 80)
    print("  Test Complete")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

