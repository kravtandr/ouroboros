
import os
import sys
import logging

# Mock env
os.environ["LOCAL_LLM_URL"] = "http://localhost:1234/v1"

# Add repo root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.DEBUG)

def test_control():
    try:
        from ouroboros.llm import local_router
        print(f"Router loaded: {local_router}")
        print(f"Has _local_url: {hasattr(local_router, '_local_url')}")
        if hasattr(local_router, '_local_url'):
             print(f"URL: {local_router._local_url()}")
        
        print(f"Available: {local_router.is_available()}")
        
        from ouroboros.tools.control import _local_llm_status
        from ouroboros.tools.registry import ToolContext
        
        ctx = ToolContext(drive_root=".", repo_dir=".")
        status = _local_llm_status(ctx)
        print("Status output:")
        print(status)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_control()
