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

st.set_page_config(
    page_title="Personalized Bot with Memory 🧠",
    page_icon="🤖",
    layout="wide"
)

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

st.title("Personalized Bot with Memory 🧠")
# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])
# Remove the @st.cache_data decorator from test_embed()
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def display_conversation_and_get_input():

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
                            # Check if the question is related to the uploaded PDF content
                            related_to_pdf = False
                            for doc in pages:
                                if any(keyword in user_prompt.lower() for keyword in doc.page_content.lower().split()):
                                    related_to_pdf = True
                                    break

                            if related_to_pdf:
                                ai_response = llm_chain.predict(question=user_prompt, chat_history=chat_history)
                            else:
                                ai_response = "I'm sorry, but I don't have any information to answer that question as it is unrelated to the content of the uploaded PDF file."
                    except openai.OpenAIError as e:
                        st.error(f"An error occurred: {e}")
                    else:
                        memory.save_context({"input": user_prompt}, {"response": ai_response})
                        st.write(ai_response)
                        new_ai_message = {"role": "assistant", "content": ai_response}
                        st.session_state.messages.append(new_ai_message)



if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
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
                index = FAISS.from_documents(pages, embeddings)
            st.success("Embeddings done.", icon="✅")

            # Add the provided model
            llm_chain = LLMChain(
                llm=ChatOpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"),
                prompt=prompt,
            )
            
            
            display_chat_messages()

            # Display the conversation and allow user input
            display_conversation_and_get_input()

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

