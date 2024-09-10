
# 📄 Chat with PDF using Gemini

This is a **Streamlit application** that allows users to upload PDF files and ask questions about their content. The app utilizes **Google's Gemini language model** for answering questions based on the content of the PDFs. The application extracts text from the PDF, processes it, and creates a vector store using the FAISS library for efficient search.

## 🛠 Features

- Upload multiple PDF files.
- Ask questions about the content of the uploaded PDFs.
- Uses Google's **Gemini language model** for generating answers.
- Automatically processes the PDFs into searchable text chunks.
  
## 📋 Table of Contents

- [Libraries Used](#-libraries-used)
- [Helper Functions](#-helper-functions)
- [Main Function](#-main-function)
- [Output](#-output)

---

## 📚 Libraries Used

The following libraries are used in this project:

- **Streamlit**: For building the user interface and displaying results.
- **PyPDF2**: To read and extract text from PDF files.
- **LangChain**: For splitting text, embedding, creating vector stores, and handling question-answering chains.
- **FAISS**: To create and manage the vector store for efficient text search.
- **Google's Generative AI**: To generate embeddings and answers using the **Gemini language model**.

---

## 🧰 Helper Functions

### `get_pdf_text(pdf_docs)`
This function extracts the text from each page of the given PDF files and combines them into a single string.

### `get_text_chunks(text)`
It splits the input text into smaller chunks using the `RecursiveCharacterTextSplitter` from LangChain. The chunks help in creating a vector store for better search performance.

### `get_vector_store(text_chunks)`
Creates a vector store from the text chunks using **FAISS** and **GoogleGenerativeAIEmbeddings**. The vector store is saved locally as `faiss_index`.

### `get_conversation_chain()`
This function defines the conversation chain that uses **ChatGoogleGenerativeAI** with a custom prompt template to generate detailed answers based on the given context (PDF content) and user question.

### `user_input(user_question)`
Takes a user’s question, loads the FAISS vector store, and performs a similarity search on the PDF content to retrieve relevant documents. The conversation chain is then used to generate an answer to the user’s question.

---

## 🚀 Main Function

### `main()`
The `main()` function sets up the **Streamlit app** and handles user interactions. Here's a breakdown:

- **User Interface**: A text input field for users to ask questions about the PDF content.
- **PDF Upload**: Allows users to upload multiple PDF files.
- **Processing Button**: After uploading, the PDF content is processed by extracting text, splitting it into chunks, and creating a vector store.

---

## ▶️ Output
![image](https://github.com/user-attachments/assets/8edd3424-4fb3-412c-bd8a-e177c93e0e6c)

---

😊 Feel free to explore, modify, and enhance this project to suit your needs. If you encounter any issues or have questions, don't hesitate to reach out.
