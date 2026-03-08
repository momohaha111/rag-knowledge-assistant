import gradio as gr
from rag_pipeline import load_document, rag_chat

def chat_interface(message, history):
    return rag_chat(message)

with gr.Blocks(title="Local RAG Assistant") as demo:

    gr.Markdown("# 📚 Local RAG Knowledge Assistant")

    with gr.Row():

        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload document",
                file_types=[".txt", ".md"],
                type="filepath"
            )

            load_btn = gr.Button("Load document")

            status = gr.Textbox(label="Status")

        with gr.Column(scale=2):

            chatbot = gr.ChatInterface(fn=chat_interface)

    load_btn.click(
        fn=load_document,
        inputs=file_upload,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch()
