# RadioBot

RadioBot is a Streamlit chatbot app that uses Azure OpenAI (GPT) and LangChain to answer questions about the content of PDF documents regarding Radiohead. It loads all PDFs from the `pdfs` directory, creates vector embeddings using HuggingFace, and enables conversational retrieval with short-term memory.

## Features

- Conversational Q&A over your PDF documents
- Uses Azure OpenAI (GPT-3.5/4) via LangChain
- HuggingFace sentence-transformers for embeddings
- FAISS for fast vector search
- Stylish chat UI with background image

## Setup

1. **Clone the repository and add your PDFs**
    ```sh
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    # Place your PDF files in the pdfs/ directory
    ```

2. **Install requirements**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file**
    ```
    AZURE_OPENAI_DEPLOYMENT=your-deployment-name
    AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    AZURE_OPENAI_API_KEY=your-azure-openai-key
    ```

4. **Run the app**
    ```sh
    streamlit run CHatbot.py
    ```

## Usage

- Ask questions about the content of your PDFs.
- The chat history is displayed for context.
- Each new question is answered using the content of your documents.

## Notes

- The `pdfs` folder must exist and contain at least one PDF.
- The `.env` file should **not** be committed to version control.
- The background image (`Thom.png`) should be present in the project directory.

---
```