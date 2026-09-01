import re

# def route_query(query: str) -> str:
#     """Returns one of: 'calculator', 'datetime', 'retrieve'"""
#     query_lower = query.lower()
    
#     datetime_keywords = ["today", "current date", "what date", "what time", "current time"]
#     if any(keyword in query_lower for keyword in datetime_keywords):
#         return "datetime"
    
#     has_math_symbols = bool(re.search(r'[\d]+\s*[\+\-\*/]\s*[\d]+', query))
#     calc_keywords = ["calculate", "what is", "plus", "minus", "times", "multiplied", "divided"]
#     has_calc_keyword = any(keyword in query_lower for keyword in calc_keywords)
#     has_numbers = bool(re.search(r'\d', query))
    
#     if has_math_symbols or (has_calc_keyword and has_numbers):
#         return "calculator"
    
#     return "retrieve"



def route_query(query: str) -> str:
    """
    Returns one of: 'calculator', 'datetime', 'retrieve'
    """
    query_lower = query.lower()

    # Rule 1: Date/time check (check this first - simpler pattern)
    datetime_keywords = ["today", "current date", "what date", "what time", "current time"]
    if any(keyword in query_lower for keyword in datetime_keywords):
        return "datetime"

    # Rule 2: Calculator check
    # Only trust explicit math symbols (e.g. "25 + 17") - avoids false positives
    # like "what is Kimi K3" matching on "what is" + the digit in "K3"
    has_math_symbols = bool(re.search(r'\d+\s*[\+\-\*/]\s*\d+', query))

    # Narrower calc keywords - removed "what is" since it's too generic
    # and matches normal knowledge questions too
    calc_keywords = ["calculate", "plus", "minus", "multiplied by", "divided by"]
    has_calc_keyword = any(keyword in query_lower for keyword in calc_keywords)
    has_numbers = bool(re.search(r'\d', query))

    if has_math_symbols or (has_calc_keyword and has_numbers):
        return "calculator"

    # Rule 3: Default to retrieval
    return "retrieve"