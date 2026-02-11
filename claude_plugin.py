import os
import sys
import logging
import asyncio
from typing import Any

# 1. CRITICAL: Silence ALL stdout to prevent breaking MCP JSON-RPC protocol
# We redirect stdout to stderr during initialization, and only restore it for FastMCP
original_stdout = sys.stdout
sys.stdout = sys.stderr 

# Configure logging to write to stderr (or file) but NEVER stdout
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='[%(name)s] %(message)s')
logger = logging.getLogger("claude_plugin")

try:
    # 2. Initialize Core Services (Quietly)
    logger.info("Initializing High-MCP Core Services for Claude Code...")
    
    # Check if API Key is available in environment
    if "ANTHROPIC_API_KEY" in os.environ:
        logger.info("✅ ANTHROPIC_API_KEY found in environment.")
    else:
        logger.warning("⚠️ ANTHROPIC_API_KEY NOT found in environment. AI features might fail.")

    from mcp.server.fastmcp import FastMCP
    from dependencies import set_dependencies
    from ai_engine import AIEngine
    from redis_quota_manager import RedisQuotaManager
    from async_adapters import LocalQuotaManagerAsync
    from error_manager import ErrorManager
    from trust_system import TrustSystem
    import components.superpowers as superpowers

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize components
    trust_system = TrustSystem(base_dir)
    quota_manager = LocalQuotaManagerAsync(base_dir)
    ai_engine = AIEngine(quota_manager, base_dir=base_dir)
    error_manager = ErrorManager(base_dir, ai_engine)

    # Inject dependencies
    set_dependencies(
        error_manager=error_manager,
        quota_manager=quota_manager,
        ai_engine=ai_engine,
        trust_system=trust_system
    )

    # 3. Setup MCP Server
    # Restore stdout for FastMCP communication
    sys.stdout = original_stdout
    
    mcp = FastMCP("High-MCP Superpowers")

    # Load Superpowers Tools
    # Note: superpowers.setup might log to stdout if not careful, but we've configured logging to stderr
    # Check if superpowers.setup writes to stdout directly? 
    # It uses 'logger', so it should be fine if logging is configured.
    superpowers.setup(mcp=mcp)
    
    logger.info("High-MCP Superpowers Plugin Ready.")

except Exception as e:
    # If init fails, print error to stderr
    sys.stderr.write(f"Failed to initialize plugin: {e}\n")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Force stdio mode for Claude Code plugin compatibility
        asyncio.run(mcp.run_stdio_async())
    except Exception as e:
        sys.stderr.write(f"MCP Run Error: {e}\n")
        sys.exit(1)
