import json

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_tavily import TavilySearch

# config
LLM_MODEL = "qwen2.5:1.5b"
TAVILY_API_KEY = "your_api"

# llm
llm = ChatOllama(model=LLM_MODEL)

# tools
tavily = TavilySearch(tavily_api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """Search the web for latest information."""
    return str(tavily.invoke({"query": query}))

@tool
def summarize(text: str) -> str:
    """Summarize the given text."""
    prompt = f"Summarize in 3-4 lines:\n{text}"
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)

@tool
def notes(text: str) -> str:
    """Convert text into short notes."""
    prompt = f"""
Convert this into notes:
Title: <title>
Content:
- point 1
- point 2
- point 3

Text:
{text}
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)

TOOLS = {
    "web_search": web_search,
    "summarize": summarize,
    "notes": notes
}

SYSTEM_PROMPT = """
You are a tool-calling AI.

You have only these tools:
1. web_search(query)
2. summarize(text)
3. notes(text)

You must reply in ONLY one of these two valid JSON formats.

Tool call:
{"tool":"web_search","args":{"query":"your query"}}

or
{"tool":"summarize","args":{"text":"your text"}}

or
{"tool":"notes","args":{"text":"your text"}}

Final answer:
{"final_answer":"your answer"}

Rules:
- Do not use any other keys
- Do not return markdown
- Do not return explanations
- Do not wrap JSON in backticks
- Do not add extra braces
- Return exactly one JSON object
"""

def run_agent(query):
    print("\nQuery:", query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    for step in range(5):
        response = llm.invoke(messages)
        output = response.content if hasattr(response, "content") else str(response)

        print("\nLLM:", output)

        try:
            data = json.loads(output)
        except:
            print("JSON error - invalid JSON returned")
            break

        if "final_answer" in data:
            print("\nAnswer:", data["final_answer"])
            break

        elif "tool" in data and "args" in data:
            name = data["tool"]
            args = data["args"]

            print("Tool:", name)

            if name in TOOLS:
                result = TOOLS[name].invoke(args)
                print("Result:", str(result)[:200])

                messages.append({"role": "assistant", "content": output})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {result}\nNow respond using only one valid JSON object with either tool/args or final_answer."
                })
            else:
                print("Unknown tool")
                break
        else:
            print("JSON error - wrong JSON format")
            break

print(web_search.invoke({"query": "OpenAI news"})[:100])
print(summarize.invoke({"text": "AI is growing fast"}))
print(notes.invoke({"text": "AI helps automate tasks"}))

run_agent("What is the latest news on OpenAI?")
run_agent("Summarize this paragraph: AI is transforming industries rapidly.")
run_agent("Find latest news on AI agents and summarize it")
