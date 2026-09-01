# """
# Conversation memory: rewrites follow-up questions into standalone questions
# using chat history, so retrieval works correctly on things like "the second one"
# """

# from langchain_core.prompts import ChatPromptTemplate

# REWRITE_PROMPT = ChatPromptTemplate.from_template("""
# Given the conversation history and a follow-up question, rewrite the follow-up 
# question to be a standalone question that includes all necessary context.

# If the follow-up question is already standalone (doesn't reference prior context), 
# return it unchanged.

# Chat History:
# {history}

# Follow-up Question: {question}

# Standalone Question:""")


# class ConversationMemory:
#     def __init__(self, llm, max_turns=5):
#         self.llm = llm
#         self.history = []  # list of (question, answer) tuples
#         self.max_turns = max_turns

#     def _format_history(self):
#         if not self.history:
#             return "No prior conversation."
#         formatted = []
#         for q, a in self.history[-self.max_turns:]:
#             formatted.append(f"User: {q}\nAssistant: {a}")
#         return "\n".join(formatted)

#     def rewrite_query(self, question: str) -> str:
#         """Rewrites a follow-up question into a standalone one, using history"""
#         if not self.history:
#             return question  # first question, nothing to rewrite

#         chain = REWRITE_PROMPT | self.llm
#         result = chain.invoke({
#             "history": self._format_history(),
#             "question": question
#         })
#         return result.content.strip()

#     def add_turn(self, question: str, answer: str):
#         self.history.append((question, answer))

"""
Conversation memory: rewrites follow-up questions into standalone questions
using chat history, so retrieval works correctly on things like "the second one"
"""

from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and a follow-up question, rewrite the follow-up 
question to be a standalone question that includes all necessary context.

If the follow-up question is already standalone (doesn't reference prior context), 
return it unchanged.

Only output the rewritten question itself - no explanation, no extra text.

Chat History:
{history}

Follow-up Question: {question}

Standalone Question:""")


class ConversationMemory:
    def __init__(self, llm, max_turns=5):
        self.llm = llm
        self.history = []  # list of (question, answer) tuples
        self.max_turns = max_turns

    def _format_history(self):
        if not self.history:
            return "No prior conversation."
        formatted = []
        for q, a in self.history[-self.max_turns:]:
            formatted.append(f"User: {q}\nAssistant: {a}")
        return "\n".join(formatted)

    def rewrite_query(self, question: str) -> str:
        """Rewrites a follow-up question into a standalone one, using history"""
        if not self.history:
            return question  # first question, nothing to rewrite

        try:
            chain = REWRITE_PROMPT | self.llm
            result = chain.invoke({
                "history": self._format_history(),
                "question": question
            })
            rewritten = result.content.strip()

            # Safety: if rewrite is empty, too short, or clearly broken,
            # fall back to the original question instead of breaking the pipeline
            if not rewritten or len(rewritten) < 5:
                return question

            return rewritten

        except Exception as e:
            print(f"   ⚠️  Memory rewrite failed ({e}), using original question")
            return question

    def add_turn(self, question: str, answer: str):
        self.history.append((question, answer))