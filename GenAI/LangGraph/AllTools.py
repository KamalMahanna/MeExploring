from langchain_core.tools import tool
import requests
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
load_dotenv("/home/kamal/.env")


travily_tool = TavilySearch(
    max_results=10
)

@tool
def calculator(equation: str) -> float:
    """Calculate the result of a math equation."""
    return eval(equation)

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/{os.getenv("EXCHANGE_RATE_API_KEY")}/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()['conversion_rate']

@tool
def get_weather(city: str) -> str:
    """
    This function fetches the weather information for a given city
    """
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv("WEATHER_API_KEY")}'

    response = requests.get(url)

    return response.json()

tools = [travily_tool, calculator, get_conversion_factor, get_weather]