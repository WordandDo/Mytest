#!/usr/bin/env python3
"""
Verification script for unified batch allocation refactoring.

This script verifies that:
1. rag_initialization function exists and has correct signature
2. setup_rag_session is marked as deprecated but still callable
3. HttpMCPRagEnv no longer overrides allocate_resource/release_resource/cleanup
4. system_tools._sync_resource_sessions includes RAG support
"""

import sys
import os
import inspect

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def check_rag_initialization():
    """Verify rag_initialization function exists"""
    print("=" * 60)
    print("1. Checking rag_initialization function...")

    try:
        from mcp_server.rag_server import rag_initialization

        # Check signature
        sig = inspect.signature(rag_initialization)
        params = list(sig.parameters.keys())

        assert "worker_id" in params, "Missing worker_id parameter"
        assert "config_content" in params, "Missing config_content parameter"

        # Check if it's async
        assert inspect.iscoroutinefunction(rag_initialization), "Should be async function"

        print("✓ rag_initialization exists with correct signature")
        print(f"  Parameters: {params}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_setup_rag_session():
    """Verify setup_rag_session is simplified"""
    print("\n" + "=" * 60)
    print("2. Checking setup_rag_session deprecation...")

    try:
        from mcp_server.rag_server import setup_rag_session

        # Check it still exists (for backward compatibility)
        assert setup_rag_session is not None, "Function should still exist"

        # Check docstring mentions deprecated
        doc = setup_rag_session.__doc__ or ""
        assert "DEPRECATED" in doc.upper(), "Should be marked as deprecated"

        print("✓ setup_rag_session exists and marked as DEPRECATED")
        print(f"  First line of doc: {doc.split(chr(10))[1].strip() if doc else 'N/A'}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def check_rag_env_overrides():
    """Verify HttpMCPRagEnv no longer overrides key methods"""
    print("\n" + "=" * 60)
    print("3. Checking HttpMCPRagEnv method overrides...")

    try:
        from envs.http_mcp_rag_env import HttpMCPRagEnv
        from envs.http_mcp_env import HttpMCPEnv

        # Get methods defined in HttpMCPRagEnv (not inherited)
        rag_methods = set(name for name in dir(HttpMCPRagEnv)
                         if not name.startswith('_') and callable(getattr(HttpMCPRagEnv, name)))

        parent_methods = set(name for name in dir(HttpMCPEnv)
                            if not name.startswith('_') and callable(getattr(HttpMCPEnv, name)))

        # Check that these methods are NOT overridden
        should_not_override = ["allocate_resource", "release_resource", "cleanup"]

        for method_name in should_not_override:
            # Check if method is defined in child class directly
            rag_method = getattr(HttpMCPRagEnv, method_name, None)
            parent_method = getattr(HttpMCPEnv, method_name, None)

            # They should be the same object (not overridden)
            if rag_method is parent_method:
                print(f"✓ {method_name}: NOT overridden (uses parent implementation)")
            else:
                # Check if it's defined in the class itself
                if method_name in HttpMCPRagEnv.__dict__:
                    print(f"✗ {method_name}: IS overridden (should use parent)")
                    return False
                else:
                    print(f"✓ {method_name}: NOT overridden (inherited from parent)")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sync_resource_sessions():
    """Verify _sync_resource_sessions handles RAG"""
    print("\n" + "=" * 60)
    print("4. Checking _sync_resource_sessions RAG support...")

    try:
        from mcp_server.system_tools import _sync_resource_sessions
        import inspect

        # Get source code
        source = inspect.getsource(_sync_resource_sessions)

        # Check for RAG handling
        assert 'if "rag" in allocated_resources' in source, "Should check for RAG resources"
        assert "RAG_SESSIONS" in source, "Should sync to RAG_SESSIONS"
        assert "base_url" in source, "Should extract base_url"

        print("✓ _sync_resource_sessions includes RAG support")
        print("  - Checks for 'rag' in allocated_resources")
        print("  - Syncs to RAG_SESSIONS")
        print("  - Extracts base_url")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("UNIFIED BATCH ALLOCATION REFACTORING VERIFICATION")
    print("=" * 60)

    results = []

    results.append(("rag_initialization function", check_rag_initialization()))
    results.append(("setup_rag_session deprecation", check_setup_rag_session()))
    results.append(("HttpMCPRagEnv overrides removed", check_rag_env_overrides()))
    results.append(("_sync_resource_sessions RAG support", check_sync_resource_sessions()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed

    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Refactoring successful!")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please review the implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
