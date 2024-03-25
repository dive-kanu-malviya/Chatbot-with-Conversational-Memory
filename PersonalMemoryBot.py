# Import necessary modules
import re
import time
from io import BytesIO
from typing import Any, Dict, List

import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from langchain_community.chat_models import ChatOpenAI

memory = ConversationBufferMemory()
if 'index' not in st.session_state:
    st.session_state.index = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

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


# Define a function for the embeddings
def create_or_load_index(pages, api):
    if st.session_state.index is None:
        embeddings = OpenAIEmbeddings(openai_api_key=api)
        with st.spinner("It's indexing..."):
            index = FAISS.from_documents(pages, embeddings)
            st.sidebar.success("Embeddings done.", icon="✅")
        st.session_state.index = index
    return st.session_state.index


# Define a function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


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

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Enter Your Secret Key for Embeddings
    3. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)


# Define a function to display conversation and get user input
def display_conversation_and_get_input(index):
    user_prompt = st.chat_input(placeholder="Enter your query here ...")

    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
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

                            qa = RetrievalQA.from_chain_type(
                                llm=OpenAI(openai_api_key=api),
                                chain_type="map_reduce",
                                retriever=index.as_retriever(),
                            )
                            # Set up the conversational agent
                            tools = [
                                Tool(
                                    name="State of Union QA System",
                                    func=qa.run,
                                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                                )
                            ]
                            prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                                                    You have access to a single tool:"""
                            suffix = """Begin!"

                                        {chat_history}
                                        Question: {input}
                                        {agent_scratchpad}"""

                            prompt = ZeroShotAgent.create_prompt(
                                tools,
                                prefix=prefix,
                                suffix=suffix,
                                input_variables=["input", "chat_history", "agent_scratchpad"],
                            )

                            if "memory" not in st.session_state:
                                st.session_state.memory = ConversationBufferMemory(
                                    memory_key="chat_history"
                                )

                            llm_chain = LLMChain(
                                llm=ChatOpenAI(
                                    temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                                ),
                                prompt=prompt,
                            )
                            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                            agent_chain = AgentExecutor.from_agent_and_tools(
                                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                            )

                            if user_prompt:
                                ai_response = agent_chain.run(user_prompt)

                    except openai.OpenAIError as e:
                        st.error(f"An error occurred: {e}")
                    else:
                        memory.save_context({"input": user_prompt}, {"response": ai_response})
                        st.write(ai_response)
                        new_ai_message = {"role": "assistant", "content": ai_response}
                        st.session_state.messages.append(new_ai_message)


# Allow the user to upload a PDF file
uploaded_file = st.sidebar.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    if pages:

        api = st.sidebar.text_input(
            "**Enter OpenAI API Key**",
            type="password",
            placeholder="sk-",
            help="https://platform.openai.com/account/api-keys",
        )
        if api:
            try:
                # Test the embeddings and save the index in a vector database
                index = create_or_load_index(pages, api)
                # index = test_embed()
                display_chat_messages()

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
            st.sidebar.warning("Please enter an OpenAI API key to continue.")
            # Disable the chat input field
            st.session_state.messages = []
            st.empty()
