import os
from autogen import ConversableAgent, AssistantAgent
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

config_list = [
    {
        "model": "llama3:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
]


model = "gpt-4o-mini"

api_key = os.getenv("OPENAI_API_KEY")

llm_config = {
    "model": model,
    "api_key": api_key,
    "temperature": 0.0,
}

# == using the local Ollama model ==
# llm_config = {"config_list": config_list, "temeperature": 0.0}

task = """
Write a concise, engaging article about AI Agentic Architecture Patterns, covering the following points:
1. Introduction to AI Agentic Architecture Patterns
2. Benefits of using AI Agentic Architecture Patterns
3. Common AI Agentic Architecture Patterns (e.g., Reflection Pattern, Delegation Pattern, etc.)
4. Best practices for implementing AI Agentic Architecture Patterns
5. Conclusion and future trends in AI Agentic Architecture Patterns

How to use Human in loop in all patterns and how to use it effectively.
Make sure to include examples and practical applications of each pattern, as well as any potential challenges and how to overcome them.
"""

writer = AssistantAgent(
    name="Writter",
    system_message="""You are writer. you write engaging and concise articles (with title) on given topics.
    You must follow polish your writing based on the feedback you receive and give a refined version. Only return your final work without additional comments.
    """,
)

reply = writer.generate_reply(messages=[{"content": task, "role": "user"}])

critic = AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATION") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of the writer and provide constructive feedback to help improve the quality of the content.",
)

res = critic.initiate_chat(
    recipient=writer, message=task, max_turns=5, summary_method="last_msg"
)

# === Add a SEO reviewer to review the article for SEO optimization ===
SEO_reviewer = AssistantAgent(
    name="SEO-Reviewer",
    llm_config=llm_config,
    system_message="""You are an SEO reviewer, known for your ability to optimize content for search engines,
    ensuring that it ranks well and attracts organic traffic. Make sure your suggestion is consice (within 10 bullet points),
    concrete and to the point.
    Begin the review by stating your role, like 'SEO Reviewer:'.""",
)

# === Add a compliance reviewer agent to suggest compliance improvements ===
compliance_reviewer = AssistantAgent(
    name="Compliance-Reviewer",
    llm_config=llm_config,
    system_message="""You are a meta-reviewer. You provide a final review of the content,
    ensuring that all the feedback from the previous reviewers has been incorporated.
    Begin the review by stating your role, like 'Meta Reviewer:'.
    """,
)

# === Meta-reviewer to aggregate all feedback and give final suggestions ===
meta_reviewer = AssistantAgent(
    name="Meta-Reviewer",
    llm_config=llm_config,
    system_message="""You are a meta-reviewer. You provide a final review of the content,
    ensuring that all the feedback from the previous reviewers has been incorporated.
    Begin the review by stating your role, like 'Meta Reviewer:'.
    """,
)


# === Orchestrate the conversation between agents and nested chats to solve the task ===
def reflection_message(recipient, messages, sender, config):
    return f"""Review the following content.
         \n\n {recipient.chat_messages_for_summary(sender)[-1]["content"]}
         """


review_chats = [
    {
        "recipient": SEO_reviewer,
        "messages": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt": "Return review into as JSON object only:"
            "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
        },
        "max_turns": 1,
    },
    {
        "recipient": compliance_reviewer,
        "messages": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt": "Return review into as JSON object only:"
            "{'Reviewer': '', 'Review': ''}.",
        },
        "max_turns": 1,
    },
    {
        "recipient": meta_reviewer,
        "messages": "Aggregrate feedback from all reviewers and give final suggenstions on the writing",
        "max_turns": 1,
    },
]

# Register reviewers and orchestrate the conversation
critic.register_nested_chats(review_chats, trigger=writer)

# === Start the conversation ===
res = critic.initiate_chat(
    recipient=writer, message=task, max_turns=2, summary_method="last_msg"
)

# Print the final article after incorporating all feedback
print("\n\n === Final Article after incorporating all feedback ===\n\n")
print(res.summary)
