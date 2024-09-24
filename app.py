import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import fitz
import os
import io
from PIL import Image
from pydantic import BaseModel
import base64
import time 
import glob

from langchain.schema import HumanMessage 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # set google api key in the environment variable

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 1. Read the PDF files and extract the text 
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)   # read the pdf in from of list as per page
        for page in pdf_reader.pages:
            text += page.extract_text()   # extract text from the each pdf page and append to the text

    return text


# convert an image file into a Base64-encoded string.
def image2base64(image_path):
    with Image.open(image_path) as image:
        buffer=io.BytesIO()
        image.save(buffer,format=image.format)
        img_str=base64.b64encode(buffer.getvalue())
        return img_str.decode("utf-8")

#  2. Read the PDF files and extract the image
def get_pdf_img(pdf_docs):
    image_list = []
    for file_path in pdf_docs:
        pdf_file = fitz.open(file_path)  # Open the saved PDF file by its path
        page_nums = len(pdf_file)
        for i in range(page_nums):
            page_content = pdf_file[i]
            image_list.extend(page_content.get_images(full=True))


    if (len(image_list) == 0):
        img_text= []
        return img_text

    # Extract the images from the PDF files and save them
    for i, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"./extracted_images/image_{i}.png")
        print(f"images/Image {i} extracted successfully")


    # set of PNG images stored in a directory, converts them to Base64 strings, and sends them to a language model for analysis.
    cwd="./extracted_images"
    files=glob.glob(cwd+"/*.png")
    img_text= []
    for file in files:
        try:
            image_str=image2base64(file) # convert the image to base64
            response=llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type":"text","text":"please give a summary of the image provided in single sentence and extract text if present, be descriptive and smart"},
                            {"type":"image_url","image_url":
                            {
                                "url":f"data:image/jpg;base64,{image_str}"

                            },
                            },
                        ]
                    )
                ]
            )
            img_name = file.split('\\')[-1]
            img_text.append(f"{img_name}: {response.content}")

        
        except Exception as e:
            print(e)
            continue

        time.sleep(5)

    return img_text

    
# # 3. Split the text and image info into chunks
def get_text_chucnks(text, img_txt):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    
    chunks = text_splitter.split_text(text)

    chunks.extend(img_txt)

    return chunks

# # 4. Create a vector store of the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001') # Load the embeddings model

    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a vector store from the text chunks
    vector_store.save_local('faiss_index')  # Save the vector store locally




# 5. Create a conversation chain to answer the questions from the PDF files
def get_conversation_chain():
    prompt_template = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, just say, "answer is not available in the context." Don't provide a wrong answer.\n\n
        Context:\n{context}?\n  
        Question:\n{question}\n
                                     
        Answer:
        """,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=500  # maximum output length
    )
    
    # Load the question answering chain
    chain = load_qa_chain(
        model,
        chain_type="stuff",  
        prompt=prompt_template
    )

    return chain


# 5. User input to ask the question from the PDF files
def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    vector_store = FAISS.load_local('faiss_index', embedding,allow_dangerous_deserialization=True)  # Load the vector store. embedding is required to load the vector store
    
    # Perform similarity search to retrieve relevant documents
    relevant_docs = vector_store.similarity_search(user_question)
    print(relevant_docs)
    chain = get_conversation_chain()  # Load the conversation chain of number 4.

    response = chain(  # pass the relevant documents and user question to the chain
        {"input_documents": relevant_docs, "question": user_question},
    )
     # print the response to the streamlit app
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini📄")   

    # User input to ask the question from the PDF files
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        # Upload the PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Read the PDF files and extract the text from 1.
                raw_img = get_pdf_img(pdf_docs)  # Read the PDF files and extract the image.
                text_chunks = get_text_chucnks(raw_text, raw_img)  # Split the text into chunks from 2.
                get_vector_store(text_chunks)  # Create a vector store of the text chunks from 3.
                st.success("Done")



if __name__ == "__main__":
    main()
