#!/usr/bin/env python3
"""
Test script to verify H1 control client setup
Run this after ./setup.sh to check all dependencies
"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 60)
    print("  Testing H1 Control Client Setup")
    print("=" * 60)
    print()
    
    tests = [
        ("NumPy", "import numpy"),
        ("Pinocchio", "import pinocchio"),
        ("CasADi", "import casadi"),
        ("Meshcat", "import meshcat"),
        ("Matplotlib", "import matplotlib"),
        ("OpenPi Client", "from openpi_client import websocket_client_policy"),
        ("Unitree SDK", "from unitree_sdk2py.core.channel import ChannelPublisher"),
        ("Inspire SDK", "from inspire_sdkpy import inspire_hand_defaut"),
        ("Local Utils", "from utils import WeightedMovingFilter"),
        ("Local Robot Control", "from robot_control import H1_2_ArmIK, H1_2_ArmController"),
    ]
    
    failed = []
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:20s} OK")
        except ImportError as e:
            print(f"❌ {name:20s} FAILED - {e}")
            failed.append(name)
        except Exception as e:
            print(f"⚠️  {name:20s} ERROR - {e}")
            failed.append(name)
    
    print()
    print("=" * 60)
    
    if not failed:
        print("✅ All tests passed! Setup is complete.")
        print()
        print("Next steps:")
        print("  1. Start policy server:")
        print("     uv run ../../scripts/mock_policy_server.py --port 8000")
        print()
        print("  2. Run H1 client:")
        print("     python h1_remote_client.py --server-host localhost")
        return 0
    else:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"   - {name}")
        print()
        print("Please run ./setup.sh again or install missing packages manually.")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())


