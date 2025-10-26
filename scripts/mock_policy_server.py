#!/usr/bin/env python3
"""
Mock policy server for testing OpenPi client connections.
Returns hardcoded action chunks without loading any actual models.
Perfect for validating network connectivity and client setup.
"""

import dataclasses
import enum
import logging
import socket
from typing import Any

import numpy as np
import tyro

from openpi.serving import websocket_policy_server

logger = logging.getLogger(__name__)


def get_network_ip():
    """Get the actual network IP address (not localhost)"""
    try:
        # Create a socket to find the actual network interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to hostname-based lookup
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            # Filter out localhost addresses
            if ip.startswith("127."):
                return "unknown"
            return ip
        except:
            return "unknown"


class EnvMode(enum.Enum):
    """Supported robot environments."""

    ALOHA = "aloha"
    DROID = "droid"
    LIBERO = "libero"


# Predefined action patterns for each command
ACTION_PATTERNS = {
    # Left arm actions
    "raise left arm": lambda: np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Move up
    "left arm up": lambda: np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "thumbs up left": lambda: np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]),  # Rotate
    "wave left": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),  # Oscillate
    "left hand down": lambda: np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Move down
    "lower left": lambda: np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    
    # Right arm actions (mirrors left but different indices)
    "raise right arm": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
    "right arm up": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
    "thumbs up right": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "wave right": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]),
    "right hand down": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]),
    "lower right": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]),
    
    # Gripper actions
    "open gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "close gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    "open left gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "close left gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    "open right gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "close right gripper": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    
    # Generic actions
    "pick": lambda: np.array([0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, -1.0]),  # Down + close
    "place": lambda: np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0]),  # Up + open
    "home": lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Neutral
}


class MockPolicy:
    """Mock policy that returns hardcoded action patterns."""
    
    def __init__(self, env: EnvMode, action_dim: int = 8, action_horizon: int = 10):
        self.env = env
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.metadata = {
            "model_type": "mock",
            "env": env.value,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "description": "Mock policy for testing connectivity",
        }
        # State for demo sequence (cycles through phases)
        self.demo_phase = 0  # 0=open, 1=raise, 2=close, 3=lower
        self.demo_chunks_in_phase = 0  # Count chunks in current phase
        self.chunks_per_phase = 3  # Number of action chunks per phase
        logger.info(f"Initialized mock policy for {env.value}")
    
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Generate mock action chunk based on the prompt."""
        prompt = observation.get("prompt", "").lower()
        
        logger.info(f"Mock inference for prompt: '{prompt}'")
        
        # Check for special sequence patterns
        if "demo" in prompt or "sequence" in prompt or "test" in prompt:
            return self._generate_demo_sequence()
        
        # Find matching action pattern
        base_action = None
        for pattern, action_fn in ACTION_PATTERNS.items():
            if pattern in prompt:
                base_action = action_fn()
                logger.info(f"  Matched pattern: '{pattern}'")
                break
        
        # Default to small random actions if no match
        if base_action is None:
            logger.info("  No pattern match, using small random actions")
            base_action = np.random.randn(self.action_dim) * 0.01
        
        # Ensure correct dimension
        if len(base_action) != self.action_dim:
            # Pad or truncate to match action_dim
            if len(base_action) < self.action_dim:
                base_action = np.pad(base_action, (0, self.action_dim - len(base_action)))
            else:
                base_action = base_action[:self.action_dim]
        
        # For H1 robots (action_dim=51): scale actions appropriately
        if self.action_dim == 51:
            # Robot joints (0-26): Scale to [0, 0.08] range for safety
            base_action[:27] = np.clip(base_action[:27] * 0.08, 0.0, 0.08)
            # Hand joints (27-50): Scale to [200, 800] range (partial open/close)
            if np.any(base_action[27:] != 0):  # Only if hand actions exist
                base_action[27:] = np.clip((base_action[27:] + 1.0) * 400, 200, 800)
            else:
                # Default: hands open at 800
                base_action[27:] = 800
        
        # Create action chunk with smooth interpolation
        action_chunk = np.zeros((self.action_horizon, self.action_dim))
        for i in range(self.action_horizon):
            # Linearly interpolate to the target action
            alpha = (i + 1) / self.action_horizon
            action_chunk[i] = base_action * alpha
            # Add small noise for realism (scale noise appropriately)
            if self.action_dim == 51:
                # Smaller noise for robot joints
                action_chunk[i, :27] += np.random.randn(27) * 0.001
                # Larger noise for hand joints
                action_chunk[i, 27:] += np.random.randn(24) * 5.0
            else:
                action_chunk[i] += np.random.randn(self.action_dim) * 0.01
        
        # Clip to valid ranges for H1
        if self.action_dim == 51:
            action_chunk[:, :27] = np.clip(action_chunk[:, :27], 0.0, 0.1)
            action_chunk[:, 27:] = np.clip(action_chunk[:, 27:], 0.0, 1000)
        
        return {
            "actions": action_chunk,
            "policy_timing": {
                "inference_ms": 5.0,  # Mock timing
                "total_ms": 5.0,
            }
        }
    
    def _generate_demo_sequence(self) -> dict[str, Any]:
        """Generate a demo sequence that cycles through phases across multiple chunks.
        
        Each phase lasts for multiple action chunks to make movements bigger and smoother.
        Phase 0: Open hands (3 chunks)
        Phase 1: Raise arms (3 chunks)
        Phase 2: Close hands (3 chunks)
        Phase 3: Lower arms (3 chunks)
        Then repeat cycle.
        """
        if self.action_dim != 51:
            # Fallback for non-H1 robots
            logger.warning("Demo sequence only works for H1 (action_dim=51), using random actions")
            action_chunk = np.random.randn(self.action_horizon, self.action_dim) * 0.01
            return {
                "actions": action_chunk,
                "policy_timing": {"inference_ms": 5.0, "total_ms": 5.0}
            }
        
        # Determine current phase
        phase_names = ["Opening hands", "Raising arms", "Closing hands", "Lowering arms"]
        logger.info(f"  Phase {self.demo_phase}: {phase_names[self.demo_phase]} (chunk {self.demo_chunks_in_phase + 1}/{self.chunks_per_phase})")
        
        # Generate action chunk for current phase
        action_chunk = np.zeros((self.action_horizon, self.action_dim))
        
        # Calculate progress within current phase (0.0 to 1.0 across all chunks in phase)
        chunk_progress = self.demo_chunks_in_phase / self.chunks_per_phase
        next_chunk_progress = (self.demo_chunks_in_phase + 1) / self.chunks_per_phase
        
        for i in range(self.action_horizon):
            # Interpolate within this chunk
            step_progress = chunk_progress + (next_chunk_progress - chunk_progress) * i / self.action_horizon
            
            if self.demo_phase == 0:  # Open hands
                # Set only ACTUATED hand joints (skip mimic joints)
                # Left hand actuated: 27, 29, 31, 33, 35, 36
                # Right hand actuated: 39, 41, 43, 45, 47, 48
                hand_value = 500 + step_progress * 400
                # Left hand
                action_chunk[i, 27] = hand_value  # index_1
                action_chunk[i, 29] = hand_value  # little_1 (pinkie)
                action_chunk[i, 31] = hand_value  # middle_1
                action_chunk[i, 33] = hand_value  # ring_1
                action_chunk[i, 35] = hand_value  # thumb_1 (rotation)
                action_chunk[i, 36] = hand_value  # thumb_2 (bend)
                # Right hand
                action_chunk[i, 39] = hand_value  # index_1
                action_chunk[i, 41] = hand_value  # little_1 (pinkie)
                action_chunk[i, 43] = hand_value  # middle_1
                action_chunk[i, 45] = hand_value  # ring_1
                action_chunk[i, 47] = hand_value  # thumb_1 (rotation)
                action_chunk[i, 48] = hand_value  # thumb_2 (bend)
                
            elif self.demo_phase == 1:  # Raise arms
                # Keep hands open
                hand_value = 900
                action_chunk[i, 27] = hand_value
                action_chunk[i, 29] = hand_value
                action_chunk[i, 31] = hand_value
                action_chunk[i, 33] = hand_value
                action_chunk[i, 35] = hand_value
                action_chunk[i, 36] = hand_value
                action_chunk[i, 39] = hand_value
                action_chunk[i, 41] = hand_value
                action_chunk[i, 43] = hand_value
                action_chunk[i, 45] = hand_value
                action_chunk[i, 47] = hand_value
                action_chunk[i, 48] = hand_value
                # Raise arms - bigger movement (0.3 radians ~ 17 degrees)
                action_chunk[i, :27] = step_progress * 0.3
                
            elif self.demo_phase == 2:  # Close hands
                action_chunk[i, :27] = 0.3  # Keep arms raised
                # Close hands: 900 to 100
                hand_value = 900 - step_progress * 800
                action_chunk[i, 27] = hand_value
                action_chunk[i, 29] = hand_value
                action_chunk[i, 31] = hand_value
                action_chunk[i, 33] = hand_value
                action_chunk[i, 35] = hand_value
                action_chunk[i, 36] = hand_value
                action_chunk[i, 39] = hand_value
                action_chunk[i, 41] = hand_value
                action_chunk[i, 43] = hand_value
                action_chunk[i, 45] = hand_value
                action_chunk[i, 47] = hand_value
                action_chunk[i, 48] = hand_value
                
            elif self.demo_phase == 3:  # Lower arms
                # Keep hands closed
                hand_value = 100
                action_chunk[i, 27] = hand_value
                action_chunk[i, 29] = hand_value
                action_chunk[i, 31] = hand_value
                action_chunk[i, 33] = hand_value
                action_chunk[i, 35] = hand_value
                action_chunk[i, 36] = hand_value
                action_chunk[i, 39] = hand_value
                action_chunk[i, 41] = hand_value
                action_chunk[i, 43] = hand_value
                action_chunk[i, 45] = hand_value
                action_chunk[i, 47] = hand_value
                action_chunk[i, 48] = hand_value
                # Lower arms back to 0
                action_chunk[i, :27] = 0.3 - step_progress * 0.3
        
        # Clip to valid ranges (increased for more visible movement)
        action_chunk[:, :27] = np.clip(action_chunk[:, :27], 0.0, 0.5)
        action_chunk[:, 27:51] = np.clip(action_chunk[:, 27:51], 0.0, 1000)
        
        # Log first action for debugging
        logger.info(f"  Hand values: L_pinkie={action_chunk[0,29]:.0f}, L_index={action_chunk[0,27]:.0f}, R_pinkie={action_chunk[0,41]:.0f}, R_index={action_chunk[0,39]:.0f}")
        
        # Update phase counter
        self.demo_chunks_in_phase += 1
        if self.demo_chunks_in_phase >= self.chunks_per_phase:
            # Move to next phase
            self.demo_chunks_in_phase = 0
            self.demo_phase = (self.demo_phase + 1) % 4
            logger.info(f"  → Moving to next phase: {phase_names[self.demo_phase]}")
        
        return {
            "actions": action_chunk,
            "policy_timing": {"inference_ms": 5.0, "total_ms": 5.0}
        }


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    
    # Environment type (determines action dimensions)
    env: EnvMode = EnvMode.DROID
    
    # Action space dimension (default: 51 for H1 humanoid)
    action_dim: int = 51
    
    # Action horizon (number of actions per chunk, default: 50 for H1)
    action_horizon: int = 50
    
    # Port to serve on
    port: int = 5006
    
    # Verbose logging
    verbose: bool = False


# Default configurations for each environment
ENV_CONFIGS = {
    EnvMode.DROID: {"action_dim": 8, "action_horizon": 10},
    EnvMode.ALOHA: {"action_dim": 14, "action_horizon": 50},
    EnvMode.LIBERO: {"action_dim": 8, "action_horizon": 10},
}


def main(args: Args) -> None:
    """Run the mock policy server."""
    
    # Use environment-specific defaults if explicitly requested and not overridden
    # Note: Default is now H1 (51, 50), so only apply env configs if user changed defaults
    if args.env in ENV_CONFIGS and (args.action_dim != 51 or args.action_horizon != 50):
        # User may have specified an environment, try to use those defaults
        config = ENV_CONFIGS[args.env]
        if args.action_dim == 51:  # User didn't override action_dim, use env default
            args.action_dim = config["action_dim"]
        if args.action_horizon == 50:  # User didn't override action_horizon, use env default
            args.action_horizon = config["action_horizon"]
    
    print("=" * 70)
    print("  Mock OpenPi Policy Server")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Environment: {args.env.value}")
    print(f"  Action Dim: {args.action_dim} (H1-2: robot 27 + hands 24)" if args.action_dim == 51 else f"  Action Dim: {args.action_dim}")
    print(f"  Action Horizon: {args.action_horizon}")
    print(f"  Port: {args.port}")
    
    # Create mock policy
    policy = MockPolicy(args.env, args.action_dim, args.action_horizon)
    
    # Get network info
    hostname = socket.gethostname()
    local_ip = get_network_ip()
    
    print(f"\nNetwork Info:")
    print(f"  Hostname: {hostname}")
    print(f"  Network IP: {local_ip}")
    print(f"  Listening on: 0.0.0.0:{args.port}")
    print(f"\n  → Clients should connect to: {local_ip}:{args.port}")
    if local_ip == "unknown":
        print(f"     (or find your IP with: ip addr show | grep 'inet ')")
    
    print(f"\nSupported action patterns:")
    patterns = list(ACTION_PATTERNS.keys())
    for i in range(0, len(patterns), 3):
        row = patterns[i:i+3]
        print(f"  {', '.join(row)}")
    
    print(f"\nConnect your client with:")
    print(f"  python h1_remote_client.py \\")
    print(f"      --server-host {local_ip} \\")
    print(f"      --server-port {args.port} \\")
    print(f"      --prompt \"demo\"")
    print("\n" + "=" * 70)
    print("  🚀 Starting server...")
    print("     Waiting for client connections...")
    print("=" * 70 + "\n")
    
    # Create and start server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  Shutting down server...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main(tyro.cli(Args))

