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



def generate_report(prompt: str) -> str:
    agent = build_agent()

    inputs = {
        "messages": [{"role": "user", "content": prompt}]
    }

    final_text = ""

    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_update in chunk.items():

            if "messages" not in node_update:
                continue

            for msg in node_update["messages"]:

                # Agent decides to call a tool
                if getattr(msg, "tool_calls", None):
                    for tool_call in msg.tool_calls:
                        print(f"🔄 [Agent] Tool: '{tool_call['name']}'")

                # Tool completed
                elif msg.type == "tool":
                    print(f"✅ [Tool] '{msg.name}' done.")

                # Save final AI response, but don't print it
                elif msg.type == "ai":
                    content = msg.content

                    if isinstance(content, list):
                        text_parts = [
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict)
                        ]
                        text = "\n".join(text_parts)
                    else:
                        text = str(content)

                    if text.strip():
                        final_text = text

    return final_text
