# Hand Control and Trajectory Saving

## What is Currently Saved in Trajectories

### Current Behavior

In `h1_execution_client.py`, the `get_current_state()` method (lines 521-542) returns the robot state that gets saved in trajectories:

```python
def get_current_state(self) -> np.ndarray:
    # Get arm joint positions (14 DOF) - MEASURED positions
    arm_q = self.robot.get_current_dual_arm_q()
    
    if self.include_hands:
        # Append current hand command values (NOT measured positions!)
        return np.concatenate([
            arm_q,                          # Measured arm positions
            self.robot.left_hand_gesture,   # Commanded hand positions
            self.robot.right_hand_gesture   # Commanded hand positions
        ])
    else:
        return arm_q
```

**Key Point**: The saved trajectories contain:
- **Arms (14 DOF)**: MEASURED joint positions (actual feedback from robot)
- **Hands (12 DOF)**: COMMANDED values (what was sent to hands, not actual positions)

### Why Commanded vs Measured?

The Inspire hands **do not provide position feedback**. The hand bridges only accept commands; they don't report back the actual finger positions. This means:

1. We send a command like `[1000, 1000, 1000, 1000, 1000, 1000]` (fully closed)
2. The hands receive and execute this command with some delay
3. We have **no way to know** when the hands actually reached that position
4. We save the **command** in the trajectory, not the actual position

### Implications for Training

This creates a **temporal misalignment** between saved states and actual robot states:

```
Time    Arm Position    Hand Command    Actual Hand Position
----    ------------    ------------    --------------------
t=0     measured        500             500 (aligned)
t=1     measured        1000            500 (delayed!)
t=2     measured        1000            750 (still moving)
t=3     measured        1000            1000 (finally aligned)
```

The saved trajectory shows `hand=1000` at t=1, but the actual hand is still at `500` with 1.5-3 second latency.

## Observed Latency Issues

### Symptoms
- Policy sends hand close command
- Hands don't respond for 1.5-3 seconds
- Delay is intermittent (sometimes fast, sometimes slow)

### Possible Causes

1. **Network Latency**
   - Hands communicate over Ethernet
   - Network congestion or packet loss could cause delays
   - Solution: Check network with `ping` and monitor bandwidth

2. **Hand Controller Buffering**
   - Hand firmware might buffer commands
   - Multiple commands sent rapidly might queue up
   - Solution: Test with slower command rates

3. **UDP/TCP Protocol Issues**
   - If using UDP: packets might be lost/reordered
   - If using TCP: congestion control might add delays
   - Solution: Check hand_bridge.py protocol implementation

4. **Command Rate Mismatch**
   - Sending commands at 30Hz but hands update slower
   - Commands might be discarded if sent too fast
   - Solution: Test different command frequencies

5. **Hand Firmware Load**
   - Hands might be processing previous commands
   - Heavy load could delay new command execution
   - Solution: Simplify commands, reduce update rate

## Testing Hand Latency

Use the provided test script:

```bash
# Run all tests
python utils/test_hand_latency.py

# Run specific test
python utils/test_hand_latency.py --test step    # Step response
python utils/test_hand_latency.py --test freq    # Frequency test
python utils/test_hand_latency.py --test buffer  # Buffering test
python utils/test_hand_latency.py --test bridge  # Bridge communication
```

### Test 1: Step Response
Measures delay from command sent to hand motion start. Helps identify baseline latency.

### Test 2: Command Frequency
Sends commands at 30Hz (same as execution). Checks if high-frequency commands cause issues.

### Test 3: Command Buffering
Sends multiple commands rapidly. Checks if commands get buffered or dropped.

### Test 4: Bridge Communication
Measures raw communication time to hand bridges. Identifies network issues.

## Recommendations

### Short-term Fixes

1. **Add Hand Command Timing Logs**
   ```python
   # In execute_action_chunk()
   logger.debug(f"Sending hand command: {left_hand[0]:.0f} at {time.time()}")
   ```

2. **Reduce Hand Command Rate**
   - Only send hand commands when they change significantly
   - Check if `abs(new_value - old_value) > threshold` before sending

3. **Add Hand Command Smoothing**
   - Interpolate hand commands over multiple timesteps
   - Avoids sudden jumps that might overwhelm the controller

### Medium-term Solutions

1. **Implement Hand Position Estimation**
   ```python
   # Track commanded positions and estimate actual positions
   self.estimated_hand_position = ...
   # Use exponential moving average or kinematic model
   ```

2. **Add Velocity-Based Hand Control**
   - Instead of position commands, send velocity commands
   - Might be more responsive for continuous motion

3. **Network Diagnostics**
   - Monitor packet loss to hand IPs
   - Use dedicated network interface for hands
   - Consider wired connection if using WiFi

### Long-term Solutions

1. **Hand Firmware with Position Feedback**
   - Modify/request hand firmware that reports positions
   - Would enable closed-loop hand control
   - Could measure actual latency accurately

2. **Synchronized Command Protocol**
   - Timestamp commands when sent
   - Hands report when they start executing
   - Allows precise latency measurement and compensation

3. **Trajectory Post-Processing**
   - After recording, adjust hand values based on measured latency
   - Shift hand commands backward in time to align with actual positions

## Impact on Policy Learning

### Current Situation
The policy learns from trajectories with misaligned hand data:
- Policy sees "hand closed" at frame N
- But hand was actually still opening at frame N
- This teaches incorrect temporal associations

### Severity
- **Low impact** if tasks don't require precise hand timing
- **High impact** if tasks require coordinated hand-arm motion
- **Critical** for tasks like "grasp object then lift"

### Mitigation Strategies

1. **Use Hand State as Context, Not Control**
   - Treat hands as binary (open/closed) rather than continuous
   - Less sensitive to exact timing

2. **Add Temporal Margin**
   - Wait extra frames after hand commands before continuing
   - Ensures hands reach target before next action

3. **Train with Latency Augmentation**
   - Artificially delay hand commands in simulation
   - Makes policy robust to latency

## Debugging Checklist

When experiencing hand latency issues:

- [ ] Run `test_hand_latency.py` to measure actual delays
- [ ] Check network latency: `ping <hand_ip>`
- [ ] Monitor CPU load on robot controller
- [ ] Check robot_control logs for errors/warnings
- [ ] Verify hand bridge is running properly
- [ ] Test with different command frequencies
- [ ] Try single hand vs both hands simultaneously
- [ ] Check if issue is position-dependent (some poses work, others don't)
- [ ] Verify hand power and connections
- [ ] Test hands in isolation (without arm motion)

## Related Files

- `h1_execution_client.py`: Main execution script (saves trajectories)
- `robot_control/robot_arm.py`: Robot controller with hand bridges
- `robot_control/hand_bridge.py`: Hand communication layer
- `utils/test_hand_latency.py`: Latency testing script (this file)
- `utils/episode_writer_hdf5.py`: Trajectory saving logic

