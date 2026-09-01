"""
Simple tools the agent can call: calculator and datetime
"""

import re
from datetime import datetime

def calculator_tool(query: str) -> str:
    """
    Extracts and evaluates a simple math expression from the query.
    Only handles + - * / for safety (no eval() on raw user input!)
    """
    # Extract expression like "25 + 17"
    match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)', query)
    
    if not match:
        return "I couldn't find a clear math expression to calculate."
    
    num1, operator, num2 = match.groups()
    num1, num2 = float(num1), float(num2)
    
    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 == 0:
            return "Cannot divide by zero."
        result = num1 / num2
    
    # Clean formatting (no unnecessary .0)
    if result == int(result):
        result = int(result)
    
    return f"The answer is {result}"


def datetime_tool(query: str) -> str:
    """Returns current date/time info"""
    now = datetime.now()
    
    if "time" in query.lower():
        return f"The current time is {now.strftime('%I:%M %p')}"
    else:
        return f"Today's date is {now.strftime('%B %d, %Y')}"


if __name__ == "__main__":
    # Quick tests
    print(calculator_tool("What is 25 + 17?"))
    print(calculator_tool("Calculate 100 / 4"))
    print(datetime_tool("What's today's date?"))
    print(datetime_tool("What time is it?"))