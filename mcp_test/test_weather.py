from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool()
async def get_weather(location: str):
    """获取指定城市（包含“市”字）的天气
    Args:
        location (str): 城市名称，必须包含“市”字
    Returns:
        str: 城市的天气描述
    """
    return f"{location}的天气是晴朗的"

if __name__ == "__main__":
    mcp.run("stdio")
