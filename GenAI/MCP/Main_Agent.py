from langchain_mcp_adapters.client import MultiServerMCPClient, Connection
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client = MultiServerMCPClient(
        {"math": Connection(command="python", args=["mathserver.py"], transport="stdio")}
    )
    
    tools = await client.get_tools()
    model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    agent = create_react_agent(
        model,tools
    )
    
    maths_response = await agent.ainvoke({"messages":[{"role":"user", "content":"what is 44455 multiplied with 469696"}]})
    
    print(maths_response)
    print(maths_response["content"])
    
asyncio.run(main()) 