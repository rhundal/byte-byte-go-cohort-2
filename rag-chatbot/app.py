# streamlit_app.py

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import lancedb
from langchain_community.llms import Ollama


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_TEMPLATE = """
You are a Customer Support Chatbot. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I don't know based on the retrieved documents."

CONTEXT:
{context}

USER QUESTION:
{question}
"""

prompt = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["context", "question"]
)

# -----------------------------
# LLM
# -----------------------------
llm = Ollama(model="gemma3:1b", temperature=0.1)

# -----------------------------
# Load embeddings
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# -----------------------------
# Connect to LanceDB
# -----------------------------
db = lancedb.connect("data/lancedb")

vectorstore = LanceDB(
    connection=db,
    table_name="rag_table",
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# -----------------------------
# Build RAG Chain
# -----------------------------
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False
)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="📚 Customer Support Chatbot", page_icon="🤖")

st.title("📚 Customer Support Chatbot")
st.write("Ask me anything about our policies, shipping, returns, and more!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    # Get response from chain
    try:
        response = chain.invoke({"question": user_input})
        answer = response["answer"]
    except Exception as e:
        answer = f"Error: {str(e)}"

    # Update chat history
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", answer))

# Display chat history
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
