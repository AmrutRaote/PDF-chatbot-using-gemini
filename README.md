# 📄 Chat with PDF using Gemini

This **Streamlit application** allows users to upload PDF files and ask questions about their content. It leverages **Google's Gemini language model** to provide accurate answers based on the content of the PDFs. The app extracts both text and images from PDFs, processes them, and creates a vector store using the FAISS library for efficient content search. **If a user’s query is related to an image within the PDF, the corresponding image will be displayed alongside the answer**, enhancing the user experience by providing visual context to the information.

## 🛠 Features

- Upload multiple PDF files and process their content.
- Extract both text and images from the PDF files.
- Uses **Google's Gemini language model** to answer user questions.
- Automatically splits PDF content into searchable text chunks and image summaries.
- Efficiently searches through PDF content using **FAISS**.
- **Dynamic Image Display**: If a user’s query is related to an image within the PDF, the corresponding image will be displayed alongside the answer.

## 📋 Table of Contents

- [Libraries Used](#-libraries-used)
- [Helper Functions](#-helper-functions)
- [Main Function](#-main-function)
- [Output](#-output)

---

## 📚 Libraries Used

The following libraries are utilized in the project:

- **Streamlit**: For creating the user interface and displaying results.
- **PyPDF2**: To extract text from PDF files.
- **LangChain**: For text splitting, embedding, creating vector stores, and handling question-answering chains.
- **FAISS**: To build and manage the vector store for fast text and image search.
- **Google's Generative AI (Gemini)**: For generating embeddings and answering questions using the **Gemini language model**.
- **PyMuPDF (Fitz)**: To extract images from PDFs.
- **Pillow**: For image processing and conversion.
- **Pydantic**: For data validation and settings management.
- **Dotenv**: For loading environment variables.

---

## 🧰 Helper Functions

### `get_pdf_text(pdf_docs)`
This function extracts text from all pages of the given PDF files. It reads each page of the PDF and appends the extracted text into a single string for later processing.

### `get_pdf_img(pdf_docs)`
This function extracts images from the given PDF files. It uses **PyMuPDF** to locate images on each page of the PDF, saves them as PNG files, and converts these images into Base64 strings for analysis by the **Gemini language model**, which provides a summary for each image.

### `get_text_chunks(text, img_txt)`
Splits the input text and image summaries into smaller chunks using the `RecursiveCharacterTextSplitter` from LangChain. These chunks are used to create a vector store for efficient searching.

### `get_vector_store(text_chunks)`
Creates a vector store using **FAISS** and **GoogleGenerativeAIEmbeddings**. The vector store is saved locally for subsequent searches.

### `get_conversation_chain()`
Defines the conversation chain using **ChatGoogleGenerativeAI** with a custom prompt template to generate detailed answers based on the provided context (PDF content) and user questions.

### `user_input(user_question)`
Takes a user’s question, loads the FAISS vector store, and performs a similarity search on the PDF content to retrieve relevant documents. The conversation chain is then used to generate a detailed answer based on the retrieved documents.

---

## 🚀 Main Function

### `main()`
The `main()` function sets up the **Streamlit app** and manages user interactions. Here’s a breakdown:

- **User Interface**: A text input field is provided for users to ask questions about the PDF content.
- **PDF Upload**: Users can upload multiple PDF files for processing.
- **Processing Button**: After the user uploads the PDF files, the content is processed by extracting text, images, and generating text chunks. The app then creates a vector store using the FAISS library to enable efficient searches.

---

## ▶️ Output
![image](https://github.com/user-attachments/assets/8edd3424-4fb3-412c-bd8a-e177c93e0e6c)
---
![image](https://github.com/user-attachments/assets/42d2a3cf-d49a-4eb2-99a9-c1d292a2bfd6)


---

😊 Feel free to explore, modify, and enhance this project to suit your needs. If you encounter any issues or have questions, don't hesitate to reach out.
