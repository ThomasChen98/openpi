#!/usr/bin/env python3
"""
Quick test script to verify connection to policy server.
This will connect and send a single test observation.
"""

import sys
import numpy as np

try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("ERROR: openpi_client not installed!")
    print("Install with: cd ../../packages/openpi-client && pip install -e .")
    sys.exit(1)

def test_connection(host: str, port: int):
    print("=" * 70)
    print("  Testing Policy Server Connection")
    print("=" * 70)
    print(f"\nConnecting to: {host}:{port}")
    
    try:
        # Connect to server
        client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port
        )
        print("✓ Connected successfully!")
        
        # Get metadata
        metadata = client.get_server_metadata()
        print(f"\nServer Metadata:")
        print(f"  Action dim: {metadata.get('action_dim', 'N/A')}")
        print(f"  Action horizon: {metadata.get('action_horizon', 'N/A')}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        
        # Create dummy observation
        print(f"\nSending test observation...")
        observation = {
            "image": {
                "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "state": np.zeros(14),  # 14 arm joints
            "prompt": "test connection"
        }
        
        # Query policy
        result = client.infer(observation)
        actions = result["actions"]
        
        print(f"✓ Received action chunk!")
        print(f"  Shape: {actions.shape}")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        if "server_timing" in result:
            timing = result["server_timing"]
            print(f"\nServer Timing:")
            print(f"  Inference: {timing.get('infer_ms', 'N/A'):.2f} ms")
        
        print(f"\n{'=' * 70}")
        print("✓ Connection test PASSED!")
        print(f"{'=' * 70}\n")
        return True
        
    except ConnectionRefusedError:
        print(f"\n✗ Connection REFUSED")
        print(f"  Make sure server is running:")
        print(f"  uv run scripts/mock_policy_server.py --port {port}")
        return False
        
    except Exception as e:
        print(f"\n✗ Connection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test policy server connection")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Policy server hostname or IP")
    parser.add_argument("--port", type=int, default=5006,
                       help="Policy server port")
    args = parser.parse_args()
    
    success = test_connection(args.host, args.port)
    sys.exit(0 if success else 1)




