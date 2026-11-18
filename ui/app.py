# app.py
"""Landing screen + navigation. Run from the ui/ folder: python app.py
(or from the repo root: python ui/app.py)"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch; Grad-CAM backward() on MPS

import gradio as gr

import core
import ui_retrieval
import ui_zeroshot


# let Gradio serve dataset images and models outside ui/
gr.set_static_paths(paths=[core.ROOT])


def show(idx):
    """Return visibility updates for [landing, retrieval, zeroshot]."""
    return tuple(gr.update(visible=(i == idx)) for i in range(3))


with gr.Blocks(title="Nep-CLIP", theme=gr.themes.Soft()) as demo:

    # ---- Landing ----
    with gr.Column(visible=True) as landing_screen:
        gr.Markdown("# 🇳🇵 Nep-CLIP")
        gr.Markdown("### Choose a task")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🔍 Image Retrieval\nSearch a collection of images by text or example image.")
                go_retrieval = gr.Button("Image Retrieval", variant="primary")
            with gr.Column():
                gr.Markdown("#### 🏷️ Zero-shot Classification\nClassify an image against candidate labels you provide.")
                go_zeroshot = gr.Button("Zero-shot Classification", variant="primary")
    # ---- Task screens (each module builds its own) ----
    retrieval_screen, back_r = ui_retrieval.build()
    zeroshot_screen, back_z = ui_zeroshot.build()

    # ---- Navigation ----
    screens = [landing_screen, retrieval_screen, zeroshot_screen]
    go_retrieval.click(lambda: show(1), None, screens)
    go_zeroshot.click(lambda: show(2), None, screens)
    back_r.click(lambda: show(0), None, screens)
    back_z.click(lambda: show(0), None, screens)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)