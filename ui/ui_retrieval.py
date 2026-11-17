# ui_retrieval.py
"""Image retrieval screen."""
import gradio as gr
import core
import os
from ui_common import dual_model_selector, wire_dual_selector, xai_compare_panel


def build():
    with gr.Column(visible=False) as screen:
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=100):
                back = gr.Button("← Back", size="sm")
            with gr.Column(scale=10):
                gr.Markdown("## 🔍 Image Retrieval — Compare")

        dd_a, dd_b, load = dual_model_selector()

        with gr.Row(equal_height=True):
            status = gr.Markdown("Select two models and click **Load both**.")
            topk = gr.Number(value=5, label="Top-k", precision=0,
                             minimum=1, maximum=50, scale=0, min_width=120)

        # --- Add images to the shared index ---
        with gr.Accordion("➕ Add images to index", open=False):
            new_images = gr.File(
                label="Select images to add",
                file_count="multiple",
                file_types=["image"],
            )
            add_btn = gr.Button("Add to index", variant="secondary")
            add_status = gr.Textbox(label="Index update status", interactive=False)

        text = gr.Textbox(label="Text query",
                          placeholder="e.g. 'सुतिरहेको कालो कुकुर'", lines=2)

        search_btn = gr.Button("Search both", variant="primary", interactive=False)

        with gr.Row():
            with gr.Column():
                status_a = gr.Textbox(label="Model A", interactive=False)
                gallery_a = gr.Gallery(label="Results A", columns=3, height="auto", object_fit="cover")
            with gr.Column():
                status_b = gr.Textbox(label="Model B", interactive=False)
                gallery_b = gr.Gallery(label="Results B", columns=3, height="auto", object_fit="cover")

        # --- add-to-index handler ---
        def _add(files):
            if not files:
                return "No images selected."
            paths = [f.name for f in files]     # gr.File -> temp file objects
            return core.add_images_to_index(paths)

        add_btn.click(_add, [new_images], [add_status])

        # --- XAI on a selected result ---
        paths_a = gr.State([])     # image paths shown in each gallery, in order
        paths_b = gr.State([])
        selected = gr.State(None)  # path of the result picked for XAI
        xai = xai_compare_panel("Click a result image above, then analyze how each word "
                                "of the text query grounds to it (both models).")

        def _search(name_a, name_b, text_, k):
            gal_a, gal_b, st_a, st_b = core.search_compare(name_a, name_b, "text", text_, None, k)
            return (gal_a, gal_b, st_a, st_b,
                    [p for p, _ in gal_a], [p for p, _ in gal_b], None,
                    "Click a result image above, then click **Analyze with XAI**.")

        def _pick(paths, evt: gr.SelectData):
            path = paths[evt.index]
            return path, f"Selected: `{os.path.basename(path)}`"

        def _explain(name_a, name_b, path, text_, method, style):
            if path is None:
                msg = "Click a result image first."
                return None, None, msg, msg
            if not text_ or not text_.strip():
                msg = "Enter a text query first."
                return None, None, msg, msg
            return core.explain_compare(name_a, name_b, path, text_, text_.split(),
                                        method, style)

        wire_dual_selector(dd_a, dd_b, load, status, search_btn)
        search_btn.click(
            _search,
            [dd_a, dd_b, text, topk],
            [gallery_a, gallery_b, status_a, status_b, paths_a, paths_b, selected, xai["info"]],
        )
        gallery_a.select(_pick, [paths_a], [selected, xai["info"]])
        gallery_b.select(_pick, [paths_b], [selected, xai["info"]])
        xai["btn"].click(
            _explain,
            [dd_a, dd_b, selected, text, xai["method"], xai["style"]],
            [xai["panel_a"], xai["panel_b"], xai["status_a"], xai["status_b"]],
        )

    return screen, back