#!/usr/bin/env python3
"""
Patch for unsloth_zoo utils.py bug where variable 'e' is referenced before assignment.

This script checks if the unsloth_zoo package has the bug and patches it automatically.
The bug is in line 42 where it references 'e' which is not defined in the exception handler.

Run this script if you encounter the error:
    NameError: name 'e' is not defined. Did you mean: 're'?
    
from unsloth_zoo.utils.py
"""

import sys
import re
from pathlib import Path


def find_unsloth_utils():
    """Find the location of unsloth_zoo/utils.py"""
    try:
        import unsloth_zoo
        utils_path = Path(unsloth_zoo.__file__).parent / "utils.py"
        return utils_path if utils_path.exists() else None
    except ImportError:
        # Try to find it manually in site-packages
        import site
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            if site_dir:
                utils_path = Path(site_dir) / "unsloth_zoo" / "utils.py"
                if utils_path.exists():
                    return utils_path
        return None


def check_needs_patch(utils_path):
    """Check if the utils.py file has the bug"""
    if not utils_path or not utils_path.exists():
        return False
    
    content = utils_path.read_text()
    
    # Check if the function already has the module handling code
    if 'isinstance(version, types.ModuleType)' in content:
        return False  # Already patched
    
    # Look for the buggy pattern: referencing 'e' without defining it
    # The bug is on line ~42: raise Exception(str(e))
    # where 'e' is not defined in the try block
    buggy_pattern = r'raise Exception\(str\(e\)\)'
    if re.search(buggy_pattern, content):
        # Check if 'except Exception as e:' is missing
        if 'except Exception as e:' not in content.split('def Version')[1].split('pass\npass')[0]:
            return True
    
    # Also need to patch if it doesn't handle module objects
    # (even if the 'e' bug is fixed)
    version_func = content.split('def Version')[1].split('pass\npass')[0]
    if 'import types' not in version_func and 'ModuleType' not in version_func:
        return True
    
    return False


def apply_patch(utils_path):
    """Apply the patch to fix the bug"""
    content = utils_path.read_text()
    
    # Replace the buggy version function
    old_code = '''def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        new_version = str(version)
        new_version = re.match(r"[0-9\.]{1,}", new_version)
        if new_version is None:
            raise Exception(str(e))
        new_version = new_version.group(0).rstrip(".")
        if new_version != version:
            new_version += ".1" # Add .1 for dev / alpha / beta / rc
        return TrueVersion(new_version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\\n"\\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass
pass'''
    
    new_code = '''def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        # Handle module objects by extracting __version__ attribute
        import types
        if isinstance(version, types.ModuleType):
            if hasattr(version, '__version__'):
                version = version.__version__
            else:
                raise Exception(f"Module {version.__name__} has no __version__ attribute")
        
        new_version = str(version)
        new_version = re.match(r"[0-9\.]{1,}", new_version)
        if new_version is None:
            raise Exception(f"Could not parse version from: {version}")
        new_version = new_version.group(0).rstrip(".")
        if new_version != str(version):
            new_version += ".1" # Add .1 for dev / alpha / beta / rc
        return TrueVersion(new_version)
    except Exception as e:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\\n"\\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]\\n"\\
            f"Error: {e}"
        )
    pass
pass'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        utils_path.write_text(content)
        return True
    
    return False


def main():
    print("Checking for unsloth_zoo bug...")
    
    utils_path = find_unsloth_utils()
    if not utils_path:
        print("❌ unsloth_zoo package not found. Is it installed?")
        return 1
    
    print(f"✓ Found unsloth_zoo/utils.py at: {utils_path}")
    
    if not check_needs_patch(utils_path):
        print("✓ No patch needed. unsloth_zoo is already fixed or doesn't have the bug.")
        return 0
    
    print("⚠ Bug detected in unsloth_zoo/utils.py")
    print("  Applying patch...")
    
    try:
        if apply_patch(utils_path):
            print("✓ Patch applied successfully!")
            print("\nYou can now run your training scripts without the NameError.")
            return 0
        else:
            print("❌ Failed to apply patch. The code structure might have changed.")
            print("   Please manually fix the bug in:", utils_path)
            print("   Change line 42 from: raise Exception(str(e))")
            print("   To: raise Exception(f'Could not parse version from: {version}')")
            print("   And add 'as e' to the except clause on line ~47")
            return 1
    except Exception as e:
        print(f"❌ Error applying patch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

