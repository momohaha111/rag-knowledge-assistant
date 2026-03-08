import gradio as gr
from rag_pipeline import load_document, generate_answer

def chat(message, history):

    try:
        response = generate_answer(message)
        return response
    except:
        return "Please upload a document first."

with gr.Blocks(title="Local RAG Knowledge Assistant") as demo:

    gr.Markdown("""
# 📚 Local RAG Knowledge Assistant

Upload a document and ask questions about it.
The system retrieves relevant context and generates answers using a local LLM.
""")

    with gr.Row():

        with gr.Column(scale=1):

            file_upload = gr.File(
                label="Upload TXT / MD file",
                file_types=[".txt", ".md"],
                type="filepath"
            )

            load_btn = gr.Button("Load Document")

            status = gr.Textbox(label="Status")

        with gr.Column(scale=2):

            chatbot = gr.ChatInterface(
                fn=chat,
                title="Document QA Chat"
            )

    load_btn.click(
        fn=load_document,
        inputs=file_upload,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch()
