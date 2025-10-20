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
        logger.info(f"Initialized mock policy for {env.value}")
    
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Generate mock action chunk based on the prompt."""
        prompt = observation.get("prompt", "").lower()
        
        logger.info(f"Mock inference for prompt: '{prompt}'")
        
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
    port: int = 8000
    
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
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "unknown"
    
    print(f"\nNetwork Info:")
    print(f"  Hostname: {hostname}")
    print(f"  Local IP: {local_ip}")
    print(f"  Listening on: 0.0.0.0:{args.port}")
    
    print(f"\nSupported action patterns:")
    patterns = list(ACTION_PATTERNS.keys())
    for i in range(0, len(patterns), 3):
        row = patterns[i:i+3]
        print(f"  {', '.join(row)}")
    
    print(f"\nStarting server...")
    print(f"\nConnect your client with:")
    print(f"  from openpi_client import websocket_client_policy")
    print(f"  client = websocket_client_policy.WebsocketClientPolicy(")
    print(f"      host='{local_ip}',  # or 'localhost' if on same machine")
    print(f"      port={args.port}")
    print(f"  )")
    print(f"  result = client.infer(observation)")
    print(f"  actions = result['actions']")
    print("\n" + "=" * 70 + "\n")
    
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
        print("\n\n👋 Shutting down server...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main(tyro.cli(Args))

