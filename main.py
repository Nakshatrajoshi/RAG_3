"""
Project 3: Agentic RAG with Hybrid Search, Memory, and Tools
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OPENROUTER_API_KEY, MODEL_NAME, BASE_URL
from router import route_query
from tools import calculator_tool, datetime_tool
from retrieval import HybridRetriever
from memory import ConversationMemory


class AgenticRAG:
    def __init__(self):
        print("🚀 Initializing Agentic RAG...\n")

        self.llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            model=MODEL_NAME,
            base_url=BASE_URL,
            temperature=0.7,
            max_tokens=512,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "RAG Project 3",
            }
        )

        self.retriever = HybridRetriever()
        self.retriever.load_and_index()

        self.memory = ConversationMemory(self.llm)

        self.answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {question}

Answer based ONLY on the context above. If the answer isn't in the context, say so.
""")

        print("✅ Agentic RAG ready!\n")

    def answer(self, question: str) -> str:
        # Step 1: Rewrite query using conversation history (if any)
        standalone_question = self.memory.rewrite_query(question)
        if standalone_question != question:
            print(f"   (Rewritten to: '{standalone_question}')")

        # Step 2: Route to the right tool/path
        route = route_query(standalone_question)
        print(f"   → Routed to: {route}")

        # Step 3: Execute based on route
        if route == "calculator":
            answer = calculator_tool(standalone_question)

        elif route == "datetime":
            answer = datetime_tool(standalone_question)

        else:  # retrieve
            chunks = self.retriever.hybrid_search(standalone_question)
            context = "\n\n".join([c.page_content for c in chunks])
            chain = self.answer_prompt | self.llm
            result = chain.invoke({"context": context, "question": standalone_question})
            answer = result.content

        # Step 4: Save this turn to memory
        self.memory.add_turn(question, answer)

        return answer


# if __name__ == "__main__":
#     agent = AgenticRAG()

#     test_questions = [
#     "What is 25 + 17?",
#     "What's today's date?",
#     "What is the main topic of the document?",   # generic - adjust based on your actual PDF
#     "Can you tell me more about that?",            # follow-up, tests memory
#     ]

#     for q in test_questions:
#         print(f"\n{'='*60}")
#         print(f"Q: {q}")
#         answer = agent.answer(q)
#         print(f"A: {answer}")


if __name__ == "__main__":
    agent = AgenticRAG()

    print("="*60)
    print("💬 Agentic RAG Chat - Ask anything about your PDFs or the web")
    print("   (calculator, date/time, or document questions)")
    print("   Type 'load <url>' to add a webpage to the knowledge base")
    print("   Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")

    while True:
        question = input("You: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue

        # Special command: load a website into the knowledge base
        if question.lower().startswith("load "):
            url = question[5:].strip()
            success = agent.retriever.load_website(url)
            if success:
                print("AI: Website added to my knowledge base — ask me about it!\n")
            continue

        answer = agent.answer(question)
        print(f"AI: {answer}\n")
