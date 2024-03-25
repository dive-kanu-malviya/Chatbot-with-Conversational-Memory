from langchain.prompts import PromptTemplate
import re
import time
from io import BytesIO
from typing import Any, Dict, List

import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import os
from langchain_community.chat_models import ChatOpenAI


# Initialize PromptTemplate and memory
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a very kind and friendly AI assistant. You are
    currently having a conversation with a human. Answer the questions
    in a kind and friendly tone with some sense of humor.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

memory = ConversationBufferMemory()

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> str:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    doc = Document(page_content=text)
    return [doc]

# Define a function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Define a function to display conversation and get user input
def display_conversation_and_get_input(index):
    user_prompt = st.chat_input()

    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Loading..."):
                    try:
                        chat_history = ""
                        memory_variables = memory.load_memory_variables({})
                        if "chat_history" in memory_variables:
                            chat_history = memory_variables["chat_history"]

                        # Check if a document is uploaded
                        if not uploaded_file:
                            ai_response = "Please upload a document first before asking a question."
                        else:
                            # Use the indexed documents to find relevant information
                            qa_chain = load_qa_chain(
                                ChatOpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"),
                                chain_type="stuff",
                                #retriever=index.as_retriever(),
                                #return_source_documents=False,  # Set to False to avoid multiple output keys
                            )
                            ai_response = qa_chain.run(input_documents=doc,question=user_prompt, chat_history=chat_history)

                    except openai.OpenAIError as e:
                        st.error(f"An error occurred: {e}")
                    else:
                        memory.save_context({"input": user_prompt}, {"response": ai_response})
                        st.write(ai_response)
                        new_ai_message = {"role": "assistant", "content": ai_response}
                        st.session_state.messages.append(new_ai_message)

# Main code starts here
# Set up the Streamlit app
st.title("🤖 Personalized Bot with Memory 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Chat with your PDF files 📜 with `Conversational Buffer Memory`  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') 
        ----
        """
)

st.markdown(
    """
    `openai`
    `langchain`
    `pypdf`
    `faiss-cpu`

    ---------
    """
)
# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Enter Your Open AI API Key for Embeddings
    3. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)

uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    # Parse uploaded PDF file and extract text content
    text_content = parse_pdf(uploaded_file)
    
    # Convert text content to a list of documents
    doc = text_to_docs(text_content)
    
    # Display OpenAI API key input field
    api = st.text_input(
        "**Enter OpenAI API Key**",
        type="password",
        placeholder="sk-",
        help="https://platform.openai.com/account/api-keys",
    )
    
    if api:
        try:
            # Create the vectorstore index
            embeddings = OpenAIEmbeddings(openai_api_key=api)
            with st.spinner("It's indexing..."):
                index = FAISS.from_documents(doc, embeddings)
            st.success("Embeddings done.", icon="✅")

            display_chat_messages()

            # Display the conversation and allow user input
            display_conversation_and_get_input(index)

        except openai.OpenAIError as e:
            if "authentication" in str(e).lower():
                st.error("Invalid API key. Please enter a valid OpenAI API key.")
                # Clear the conversation and text field
                st.session_state.messages = []
                st.empty()
            else:
                st.error("Invalid API key. Please enter a valid OpenAI API key.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter an OpenAI API key to continue.")
        # Disable the chat input field
        st.session_state.messages = []
        st.empty()

