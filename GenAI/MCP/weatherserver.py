import requests
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """get current weather details of a location

    Args:
        location (str): Name of city or country

    Returns:
        str: weather details with json object
    """
    response = requests.get(f"https://api.weatherapi.com/v1/current.json?q={location}&key={os.getenv('WEATHER_API_KEY')}")
    print("got the weather")
    return response.json()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")