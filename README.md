# Quota MCP Server

This MCP server provides access to the quota information from `quota.xlsx`.

## Setup

1.  Navigate to this directory:
    ```powershell
    cd C:\Users\HP\Documents\high-mcp
    ```
2.  Create a virtual environment (optional but recommended):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate
    ```
3.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

## Usage

Run the server using the MCP CLI or configure it in your MCP client (like Claude Desktop or Trae).

### Running via FastMCP Dev Server (for testing)
```powershell
mcp dev server.py
```

### Configuring in Claude/Trae
Add the following to your MCP configuration file:

```json
{
  "mcpServers": {
    "quota-server": {
      "command": "python",
      "args": ["C:\\Users\\HP\\Documents\\high-mcp\\server.py"]
    }
  }
}
```
*Note: Use the full path to your python executable if using a venv.*

## Tools Available
- `list_models`: List all available models.
- `get_quota`: Get quota details for a specific model.
- `get_quota_by_category`: List models in a category.

## Resources Available
- `quota://all`: Get the full quota dataset.
"# high-mcp" 
