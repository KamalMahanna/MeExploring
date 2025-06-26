from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int | float, b: int | float) -> int | float:
    """add two numbers

    Args:
        a (int | float): the first number you want to add
        b (int | float): the second number you want to add

    Returns:
        int | float: the sum of the two numbers
    """
    print(a+b)
    return a + b

@mcp.tool()
def multiply(a: int | float, b: int | float) -> int | float:
    """multiply two numbers

    Args:
        a (int | float): the first number you want to multiply
        b (int | float): the second number you want to multiply

    Returns:
        int | float: the product of the two numbers
    """
    print(a*b)
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")