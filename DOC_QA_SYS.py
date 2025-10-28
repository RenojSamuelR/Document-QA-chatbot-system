# 🧩 Install all required libraries
!pip install -q gradio transformers torch langchain langchain-community pypdf faiss-cpu sentence-transformers pydub openai-whisper PyPDF2
!apt install -y ffmpeg  # Needed for audio processing

# 🧠 Imports
import os
import tempfile
import gradio as gr
import whisper
from pydub import AudioSegment
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 🔊 Load faster Whisper model (base)
whisper_model = whisper.load_model("base")

# 🧠 Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🤖 Text2Text LLM pipeline
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

# 📄 Process PDF
def process_pdf(pdf_file, status=gr.Progress()):
    status(0, desc="📄 Reading PDF...")
    reader = PdfReader(pdf_file.name)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    status(0.3, desc="✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    status(0.6, desc="🧠 Generating embeddings...")
    docsearch = FAISS.from_texts(texts, embedding_model)

    status(1.0, desc="✅ Done!")
    return docsearch

# 🎙️ Transcribe Audio
def transcribe_audio(audio_file, status=gr.Progress()):
    try:
        status(0, desc="🔄 Converting to WAV...")
        audio = AudioSegment.from_file(audio_file.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            # Reduce audio sample rate to 16000Hz to speed up Whisper
            audio.export(tmp_wav.name, format="wav", parameters=["-ar", "16000"])

        status(0.5, desc="🧠 Transcribing with Whisper (base)...")
        result = whisper_model.transcribe(tmp_wav.name)

        status(1.0, desc="✅ Done!")
        return result["text"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ❓ Ask question from processed content
def answer_question(question, docsearch):
    if not docsearch:
        return "❌ Please process a document or audio file first."

    try:
        docs = docsearch.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the context:\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        result = llm_pipeline(prompt, max_new_tokens=150)
        return result[0]["generated_text"]
    except Exception as e:
        return f"❌ Error while answering: {str(e)}"

# 🌐 Global docsearch
docsearch_global = None

# 🔁 Combined processor
def process_files(pdf, audio, status=gr.Progress()):
    global docsearch_global

    if pdf:
        docsearch_global = process_pdf(pdf, status=status)
        return "✅ PDF processed successfully. You can now ask questions."

    if audio:
        text = transcribe_audio(audio, status=status)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        docsearch_global = FAISS.from_texts(texts, embedding_model)
        return "✅ Audio processed successfully. You can now ask questions."

    return "⚠️ Please upload a PDF or an audio file to process."

# 🎛️ Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Doc Question answering")

    with gr.Row():
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        audio_input = gr.File(label="🔊 Upload MP3 Audio", file_types=[".mp3"])

    process_btn = gr.Button("📂 Process Document/Audio")
    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="❓ Enter your question")
        ask_btn = gr.Button("🧠 Ask")

    answer_output = gr.Textbox(label="💡 Answer")

    process_btn.click(process_files, inputs=[pdf_input, audio_input], outputs=status_output)
    ask_btn.click(fn=lambda q: answer_question(q, docsearch_global),
                  inputs=question_input, outputs=answer_output)

demo.launch(debug=True)
