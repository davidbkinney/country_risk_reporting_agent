from . import rag_tools, prompts
import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

_GOOGLE_API_KEY = None

def set_google_key(key: str):
    _GOOGLE_API_KEY = key


def _get_google_key():
    return _GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")



def build_agent():
    api_key = _get_google_key()

    if not api_key:
        raise ValueError(
            "Set your Google API key using set_google_key('...') "
            "or os.environ['GOOGLE_API_KEY']"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key
    )

    return create_agent(
        model=llm,
        tools=[
            rag_tools.get_country_wiki_summary,
            rag_tools.get_worry_experience_gap,
            rag_tools.country_search_2019,
            rag_tools.field_search_2019,
            rag_tools.database_query_2019,
            rag_tools.country_search_2021,
            rag_tools.field_search_2021,
            rag_tools.database_query_2021,
            rag_tools.country_search_2023,
            rag_tools.field_search_2023,
            rag_tools.database_query_2023,
            rag_tools.country_search_mortality,
            rag_tools.get_mortality_data
        ],
        system_prompt=prompts.SYSTEM_PROMPT
    )



def generate_response(prompt: str) -> str:
    agent = build_agent()

    inputs = {
        "messages": [{"role": "user", "content": prompt}]
    }

    final_text = ""

    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            if "messages" in node_update:
                for msg in node_update["messages"]:

                    if msg.type == "ai":
                        content = msg.content
                        if isinstance(content, list):
                            first = content[0]
                            text = first["text"] if isinstance(first, dict) and "text" in first else str(first)
                        else:
                            text = content

                        if text.strip():
                            print(f"\n💬 [Agent]: {text}")
                            final_text = text

                        if getattr(msg, "tool_calls", None):
                            for tool_call in msg.tool_calls:
                                print(f"🔄 [Agent] Tool: '{tool_call['name']}'")

                    elif msg.type == "tool":
                        print(f"✅ [Tool] '{msg.name}' done.")

    return final_text