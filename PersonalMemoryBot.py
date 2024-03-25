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

from dotenv import load_dotenv

print(load_dotenv())
memory = ConversationBufferMemory()


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
# @st.cache_data
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB

    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.sidebar.success("Embeddings done.", icon="✅")
    return index


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

# st.markdown(
#     """
#     `openai`
#     `langchain`
#     `pypdf`
#     `faiss-cpu`
#
#     ---------
#     """
# )

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

# Allow the user to upload a PDF file
uploaded_file = st.sidebar.file_uploader("**Upload Your PDF File**", type=["pdf"])


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
                            # qa_chain = load_qa_chain(
                            #     ChatOpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"),
                            #     chain_type="stuff",
                            #     #retriever=index.as_retriever(),
                            #     #return_source_documents=False,  # Set to False to avoid multiple output keys
                            # )
                            # ai_response = qa_chain.run(input_documents=doc,question=user_prompt,
                            # chat_history=chat_history)
                            # Set up the question-answering system
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

                            # # Allow the user to enter a query and generate a response
                            # query = st.text_input(
                            #     "**What's on your mind?**",
                            #     placeholder="Ask me anything from {}".format(name_of_file),
                            # )

                            if user_prompt:
                                # with st.spinner(
                                #         "Generating Answer to your Query : `{}` ".format(user_prompt)
                                # ):
                                ai_response = agent_chain.run(user_prompt)

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
    if pages:
        # Allow the user to select a page and view its content
        # with st.expander("Show Page Content", expanded=False):
        #     page_sel = st.number_input(
        #         label="Select Page", min_value=1, max_value=len(pages), step=1
        #     )
        #     pages[page_sel - 1]
        # Allow the user to enter an OpenAI API key
        api = st.sidebar.text_input(
            "**Enter OpenAI API Key**",
            type="password",
            placeholder="sk-",
            help="https://platform.openai.com/account/api-keys",
        )
        if api:
            try:
                # Test the embeddings and save the index in a vector database
                index = test_embed()
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


            # # Set up the question-answering system
            # qa = RetrievalQA.from_chain_type(
            #     llm=OpenAI(openai_api_key=api),
            #     chain_type="map_reduce",
            #     retriever=index.as_retriever(),
            # )
            # # Set up the conversational agent
            # tools = [
            #     Tool(
            #         name="State of Union QA System",
            #         func=qa.run,
            #         description="Useful for when you need to answer questions about the aspects asked. Input may be a
            #         partial or fully formed question.",
            #     )
            # ]
            # prefix = """Have a conversation with a human, answering the following questions as best you can based on
            # the context and memory available.
            #             You have access to a single tool:"""
            # suffix = """Begin!"
            #
            # {chat_history}
            # Question: {input}
            # {agent_scratchpad}"""
            #
            # prompt = ZeroShotAgent.create_prompt(
            #     tools,
            #     prefix=prefix,
            #     suffix=suffix,
            #     input_variables=["input", "chat_history", "agent_scratchpad"],
            # )
            #
            # if "memory" not in st.session_state:
            #     st.session_state.memory = ConversationBufferMemory(
            #         memory_key="chat_history"
            #     )
            #
            # llm_chain = LLMChain(
            #     llm=ChatOpenAI(
            #         temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
            #     ),
            #     prompt=prompt,
            # )
            # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            # agent_chain = AgentExecutor.from_agent_and_tools(
            #     agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            # )
            #
            # # Allow the user to enter a query and generate a response
            # query = st.text_input(
            #     "**What's on your mind?**",
            #     placeholder="Ask me anything from {}".format(name_of_file),
            # )
            #
            # if query:
            #     with st.spinner(
            #             "Generating Answer to your Query : `{}` ".format(query)
            #     ):
            #         res = agent_chain.run(query)
            #         st.info(res, icon="🤖")

            # # Allow the user to view the conversation history and other information stored in the agent's memory
            # with st.expander("History/Memory"):
            #     st.session_state.memory
