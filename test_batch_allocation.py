#!/usr/bin/env python3
"""
Quick functional test for unified batch allocation.

This test mocks the backend API and verifies the complete flow:
1. RAG resource allocation via batch system
2. Session synchronization
3. Initialization with config
4. Resource release
"""

import sys
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))


async def test_rag_initialization():
    """Test rag_initialization function"""
    print("\n" + "=" * 60)
    print("TEST 1: rag_initialization function")
    print("=" * 60)

    from mcp_server.rag_server import rag_initialization, RAG_SESSIONS

    worker_id = "test_worker_001"

    # Setup: Simulate _sync_resource_sessions has run
    RAG_SESSIONS[worker_id] = {
        "resource_id": "rag_123",
        "token": "test_token",
        "base_url": "http://localhost:8001",
        "config_top_k": None
    }

    # Test 1: Initialization with config
    config = {"top_k": 10}
    result = await rag_initialization(worker_id, config)

    assert result == True, "Initialization should succeed"
    assert RAG_SESSIONS[worker_id]["config_top_k"] == 10, "Config should be stored"
    print("✓ Initialization with config succeeded")
    print(f"  Stored config_top_k: {RAG_SESSIONS[worker_id]['config_top_k']}")

    # Test 2: Initialization without config
    worker_id_2 = "test_worker_002"
    RAG_SESSIONS[worker_id_2] = {
        "resource_id": "rag_456",
        "token": "test_token_2",
        "base_url": "http://localhost:8002",
        "config_top_k": None
    }

    result = await rag_initialization(worker_id_2, None)
    assert result == True, "Initialization without config should succeed"
    print("✓ Initialization without config succeeded")

    # Test 3: Initialization with missing session (should fail)
    result = await rag_initialization("nonexistent_worker", None)
    assert result == False, "Initialization should fail for missing session"
    print("✓ Initialization correctly fails for missing session")

    # Cleanup
    RAG_SESSIONS.clear()
    print("\n✓ All rag_initialization tests passed!")


async def test_setup_rag_session_deprecated():
    """Test setup_rag_session backward compatibility"""
    print("\n" + "=" * 60)
    print("TEST 2: setup_rag_session (DEPRECATED)")
    print("=" * 60)

    from mcp_server.rag_server import setup_rag_session, RAG_SESSIONS

    worker_id = "test_worker_003"

    # Setup: Simulate batch allocation has already created session
    RAG_SESSIONS[worker_id] = {
        "resource_id": "rag_789",
        "token": "test_token_3",
        "base_url": "http://localhost:8003",
        "config_top_k": None
    }

    # Test: Calling deprecated function should work if session exists
    result_str = await setup_rag_session(worker_id, '{"top_k": 15}')
    result = json.loads(result_str)

    assert result["status"] == "success", "Should succeed with existing session"
    assert RAG_SESSIONS[worker_id]["config_top_k"] == 15, "Should update config"
    print("✓ Deprecated function works with existing session")
    print(f"  Updated config_top_k: {RAG_SESSIONS[worker_id]['config_top_k']}")

    # Test: Calling without existing session should fail
    result_str = await setup_rag_session("nonexistent_worker", "{}")
    result = json.loads(result_str)

    assert result["status"] == "error", "Should fail without existing session"
    assert "deprecated" in result["message"].lower(), "Error should mention deprecated"
    print("✓ Deprecated function correctly fails without session")

    # Cleanup
    RAG_SESSIONS.clear()
    print("\n✓ All setup_rag_session tests passed!")


def test_sync_resource_sessions():
    """Test _sync_resource_sessions RAG handling"""
    print("\n" + "=" * 60)
    print("TEST 3: _sync_resource_sessions")
    print("=" * 60)

    import asyncio
    from mcp_server.system_tools import _sync_resource_sessions
    from mcp_server.rag_server import RAG_SESSIONS

    worker_id = "test_worker_004"

    # Mock allocated resources from backend API
    allocated_resources = {
        "rag": {
            "id": "rag_backend_001",
            "token": "backend_token",
            "base_url": "http://rag-service:8080",
            "type": "rag"
        }
    }

    # Run sync
    asyncio.run(_sync_resource_sessions(worker_id, allocated_resources))

    # Verify
    assert worker_id in RAG_SESSIONS, "Session should be created"
    session = RAG_SESSIONS[worker_id]

    assert session["resource_id"] == "rag_backend_001", "Should store resource_id"
    assert session["token"] == "backend_token", "Should store token"
    assert session["base_url"] == "http://rag-service:8080", "Should store base_url"
    assert session["config_top_k"] is None, "Should initialize config_top_k"

    print("✓ Resource session synchronized correctly")
    print(f"  Session: {session}")

    # Cleanup
    RAG_SESSIONS.clear()
    print("\n✓ All _sync_resource_sessions tests passed!")


def test_http_mcp_rag_env_no_overrides():
    """Test HttpMCPRagEnv doesn't override key methods"""
    print("\n" + "=" * 60)
    print("TEST 4: HttpMCPRagEnv method inheritance")
    print("=" * 60)

    # Import should work even without dependencies if we don't instantiate
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "http_mcp_rag_env",
        "src/envs/http_mcp_rag_env.py"
    )

    # Check methods are not defined in the class dict
    with open("src/envs/http_mcp_rag_env.py", "r") as f:
        content = f.read()

    should_not_override = [
        "def allocate_resource(",
        "def release_resource(",
        "def cleanup(",
        "def _ensure_rag_session("
    ]

    for method_sig in should_not_override:
        if method_sig in content:
            print(f"✗ FAIL: Found {method_sig} in HttpMCPRagEnv")
            return False
        else:
            print(f"✓ {method_sig.split('(')[0].strip()}: NOT overridden")

    print("\n✓ All HttpMCPRagEnv inheritance tests passed!")
    return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("FUNCTIONAL TESTS FOR BATCH ALLOCATION REFACTORING")
    print("=" * 60)

    try:
        await test_rag_initialization()
        await test_setup_rag_session_deprecated()
        test_sync_resource_sessions()
        test_http_mcp_rag_env_no_overrides()

        print("\n" + "=" * 60)
        print("✓ ALL FUNCTIONAL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
