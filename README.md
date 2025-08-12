model embedding : 
- gemini-embedding-001
- https://huggingface.co/Qwen/Qwen3-Embedding-8B

"what is vision of computer science department"
"what course that available on KBK package artificial intelligence"
"explain the facilities available in the computer science department, especially laboratorium"

template = f"""Answer the question based only on the following context (+ chat history if relevant). 
    If you don't know the answer, say "I don't know". 
    If any context is irrelevant to the question, do not use it. Here is the context : 
    {context}

    The chat history is as follows:
    {chat_history}

    The question is as follows:
    {query}
    """

template = f"""You are an AI assistant answering questions based strictly on the provided context and, if present, the chat history.

Only use information that is clearly relevant to the question. Ignore unrelated or ambiguous context. If the answer cannot be determined from the information provided, respond with: "I don't know."

Context:
{context}

Chat history:
{chat_history}

Question:
{query}
"""
