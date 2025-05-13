# Interactive File Query GPT 🎈

**Interactive File Query GPT** is a Streamlit-based application that allows users to upload various file formats (PDF, PPT, TXT, Audio) and query the extracted content using a locally available AI model.

---

## Features 🚀

- **Supports multiple file formats**: PDF, PPT, TXT, and Audio files (WAV, MP3).
- **AI-powered text extraction & querying**: Uses `Ollama` for vector embeddings and document retrieval.
- **Multi-query retrieval**: Generates multiple versions of user questions to improve document search.
- **Whisper integration**: Transcribes audio files using OpenAI Whisper.
- **PDF page visualization**: Displays uploaded PDFs as images for better visualization.

---

## Installation 🛠️

### Prerequisites![resumepdf_converted_to_img](https://github.com/user-attachments/assets/10ecd3a0-cf67-4a60-85b0-06c63fea8cfa)
![recommendingquestions](https://github.com/user-attachments/assets/47005a0e-06f3-47ea-93e8-263b31b40fe8)
![ollama_models_interface](https://github.com/user-attachments/assets/7b989287-ee03-4eb6-812a-248bb213651c)
![ollama_models](https://github.com/user-attachments/assets/10a46146-aec0-4abf-8003-b2516be6cf38)
![model_loading_extracting_images](https://github.com/user-attachments/assets/e6d6214b-75fd-4f26-9cef-411ec614f569)
![interactive_file_query_gpt_resumequestionoutput](https://github.com/user-attachments/assets/338f22f2-26b4-4636-a488-335992fa1da9)
![interactive_file_query_gpt_interface](https://github.com/user-attachments/assets/7d9a53c8-84f3-4a31-8661-ff4ea4ac2e63)


Ensure you have the following installed:

- Python 3.8+
- pip
- `ffmpeg` (for audio processing)

### Clone the Repository

```sh
git clone https://github.com/your-repo/interactive-file-query-gpt.git
cd interactive-file-query-gpt
```

Install Dependencies
pip install -r requirements.txt


Usage 🏃
Run the Application

streamlit run app.py

Upload and Query Files
Upload a PDF, PPT, TXT, or Audio file.
Select a locally available AI model.
Ask questions related to the document.
View extracted text, transcriptions, or PDF pages.
Delete the vector database if needed.

Dependencies 📦


Streamlit: UI framework
Ollama: LLM-based embeddings
LangChain: Document processing & querying
pdfplumber: PDF text extraction
Whisper: Audio transcription
Chroma: Vector database
Pillow: Image processing
python-pptx: PowerPoint file handling

Troubleshooting ❓

If Whisper fails to transcribe, ensure ffmpeg is installed
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # MacOS


If Ollama models are not available, install them manually:

ollama pull nomic-embed-text
ollama pull any-other-model

![resumepdf_converted_to_img](https://github.com/user-attachments/assets/67957a68-3523-4b22-a925-4a11c60d0ff1)
![recommendingquestions](https://github.com/user-attachments/assets/b13507c5-0982-41ff-95e3-2d17f62824c8)
![ollama_models_interface](https://github.com/user-attachments/assets/0b1d3658-992e-4ba2-aab4-3315fea0ec88)
![ollama_models](https://github.com/user-attachments/assets/8396d61c-ce35-4c7e-a522-27bcb9133a16)
![model_loading_extracting_images](https://github.com/user-attachments/assets/a84ef19f-b373-4d45-b28a-f06145ef98f9)
![interactive_file_query_gpt_resumequestionoutput](https://github.com/user-attachments/assets/0d05229d-c471-4e99-ba1b-16f7b4b71aa4)
![interactive_file_query_gpt_interface](https://github.com/user-attachments/assets/ecd47430-ff2c-4eda-94ee-e99a074e8b30)

