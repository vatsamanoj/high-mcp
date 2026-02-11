# High-Availability MCP Quota Server

This server implements the Model Context Protocol (MCP) to provide high-availability AI content generation with quota management.

## Features
- **High Availability**: Automatically switches to fallback models when one is exhausted or fails.
- **Quota Management**: Real-time tracking of RPM, TPM, and RPD using Redis (via `fakeredis` for local persistence).
- **Hot-Reloading**: Automatically digests new or modified `quota_*.json` files from the `quotas/` directory without restarting.
- **Multi-Provider Support**: Supports Google Gemini, Groq, Mistral, and OpenRouter (via OpenAI-compatible API).
- **FreeAISensor**: A background "health check" service that periodically probes free models to ensure they are available and working, updating their status in real-time.
- **Key Harvester Agent**: An intelligent, browser-based agent that helps you navigate provider dashboards to "grab" API keys using computer vision and DOM analysis.
- **Persistence**: Usage data is persisted to `redis_dump.json`.
- **Standard Interface**: Fully compatible with any MCP client (Claude Desktop, Cursor, etc.).

## Setup

1. **Add Quota Files**:
   - `quotas/quota_gemini.json` (Google)
   - `quotas/quota_groq.json` (Groq Free Tier)
   - `quotas/quota_openrouter_free.json` (OpenRouter Free Tier)
   - `quotas/quota_cerebras.json` (Cerebras Free Tier)
   - `quotas/quota_sambanova.json` (SambaNova Free Tier)
   
   *Note: You must edit these files to add your actual API keys.*

## Free AI Provider Guide (2025)

The AI landscape is evolving rapidly. Here are the best high-performance free tiers currently available, which you can use with this server:

### 1. Cerebras (Inference Speed Champion)
- **Limit**: 30 RPM, 1M tokens/day.
- **Models**: Llama 3.1 8B/70B.
- **Get Key**: [inference.cerebras.ai](https://inference.cerebras.ai/)

### 2. SambaNova (Massive Models)
- **Limit**: Generous free tier (limits vary).
- **Models**: Llama 3.1 405B (!!), 70B, 8B, Qwen 2.5.
- **Get Key**: [cloud.sambanova.ai](https://cloud.sambanova.ai/)

### 3. Groq (The Original Fast AI)
- **Limit**: 30 RPM, ~14k requests/day.
- **Models**: Llama 3, Mixtral, Gemma.
- **Get Key**: [console.groq.com](https://console.groq.com/)

### 4. Google AI Studio (High Volume)
- **Limit**: 15 RPM, 1M+ tokens/day.
- **Models**: Gemini 2.0 Flash, Pro.
- **Get Key**: [aistudio.google.com](https://aistudio.google.com/)

### 5. OpenRouter (The Aggregator)
- **Limit**: Varies (many free models).
- **Models**: Access to "Free" models from various labs.
- **Get Key**: [openrouter.ai](https://openrouter.ai/)

## Auto Key Harvester

We have included a semi-autonomous agent to help you fetch keys.

1. **Run the Harvester**:
   ```bash
   python key_harvester/harvester.py
   ```
2. **Follow Instructions**:
   - Select the target provider (e.g., `groq`).
   - The script will launch a browser.
   - **Log in manually** to the dashboard.
   - The AI Agent will analyze the page, find the "Create Key" button or the key itself, and save it to the `quotas/` folder automatically.

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install playwright beautifulsoup4
   playwright install chromium
   ```

2. **Run Server**:
   ```bash
   python server.py
   ```
   (This runs the server over Stdio, suitable for integration with other tools).

## Configuration for AI Coders

To use this server with "Renowned AI Coders" like Claude Desktop, add the following configuration to your `claude_desktop_config.json`:

### Windows
File location: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "quota-server": {
      "command": "python",
      "args": [
        "C:\\Users\\HP\\Documents\\high-mcp\\server.py"
      ]
    }
  }
}
```

Make sure to use the absolute path to your `server.py`.

## Testing

You can verify the server functionality and model availability using the included scripts:

- **Verify All Models**:
  ```bash
  python verification_scripts/verify_all_models.py
  ```
  This script checks all models defined in your quota files and generates a `quota_verification_report.json`.

- **Test MCP Protocol**:
  ```bash
  python testing_scripts/test_stdio_client.py
  ```
  This script simulates an MCP client (like Claude) connecting to the server and listing available tools.

- **Test Quota Logic**:
  ```bash
  python testing_scripts/test_redis_quota.py
  ```

## Quota Management
The server uses a **Redis-based manager** (simulated with `fakeredis`) to handle quotas.

- **Configuration**: Place your quota configuration files (e.g., `quota_gemini.json`) in the `quotas/` directory.
- **Hot-Reloading**: The server watches the `quotas/` directory. Simply drop a new JSON file or edit an existing one, and the changes will be applied instantly.
- **Persistence**: Usage stats (RPM/RPD counts) are saved to `redis_dump.json` every 5 seconds.
- **Reset**: To reset usage counters, you can delete `redis_dump.json`.
