#!/usr/bin/env python3
"""
Simple test script to verify that the agentic server and work dispatcher work together.
This can be used to test the separation without running the full evaluation.
"""

import sys
import os
# Add project root to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import aiohttp
import time
import subprocess
import signal
from typing import Optional

async def test_server_health(server_url: str = "http://localhost:5000") -> bool:
    """Test if the server is healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    health_data = await response.json()
                    return health_data.get("models_initialized", False)
                return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

async def test_beam_search_request(server_url: str = "http://localhost:5000") -> bool:
    """Test a simple beam search request"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "problem": "What is 2 + 2?",
                "search_width": 2,
                "select_top_k": 1,
                "max_iterations": 5
            }
            
            print("Sending test beam search request...")
            async with session.post(f"{server_url}/beam_search", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    thoughts = result.get("thoughts", [])
                    print(f"✅ Received {len(thoughts)} thoughts from server")
                    
                    # Print first thought for debugging
                    if thoughts:
                        first_thought = thoughts[0]
                        print(f"First thought steps: {first_thought.get('steps', [])}")
                        print(f"First thought scores: {first_thought.get('scores', [])}")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Server error {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def start_server_subprocess() -> Optional[subprocess.Popen]:
    """Start the agentic server as a subprocess"""
    try:
        server_script = os.path.join(os.path.dirname(__file__), "agentic_server.py")
        process = subprocess.Popen([sys.executable, server_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

async def wait_for_server_startup(max_wait_time: int = 60) -> bool:
    """Wait for the server to start up and be ready"""
    print("Waiting for server to start up...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if await test_server_health():
            return True
        await asyncio.sleep(2)
    
    return False

async def main():
    """Main test function"""
    print("="*60)
    print("TESTING SEPARATED BEAM SEARCH COMPONENTS")
    print("="*60)
    
    server_url = "http://localhost:5000"
    
    # First, check if server is already running
    print("Checking if server is already running...")
    if await test_server_health(server_url):
        print("✅ Server is already running and healthy")
        server_process = None
    else:
        print("Server not running. Starting server subprocess...")
        server_process = start_server_subprocess()
        
        if server_process is None:
            print("❌ Failed to start server")
            return False
        
        # Wait for server to be ready
        if not await wait_for_server_startup():
            print("❌ Server failed to start within timeout")
            if server_process:
                server_process.terminate()
            return False
        
        print("✅ Server started successfully")
    
    try:
        # Test the beam search endpoint
        success = await test_beam_search_request(server_url)
        
        if success:
            print("\n✅ All tests passed! The separated components work correctly.")
            print("\nTo run the full evaluation:")
            print("1. Start the agentic server: python src/examples/agentic_server.py")
            print("2. Run the work dispatcher: python src/examples/work_dispatcher.py")
        else:
            print("\n❌ Tests failed!")
            
        return success
        
    finally:
        # Clean up subprocess if we started it
        if server_process:
            print("\nStopping server subprocess...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
