import gradio as gr
from sd import stl_list, img_size_opt_dict, generate_images

with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Stable Diffusion - Text Inversion and additional guidence</h1>")
    gr.HTML("<h4 align = 'center'> Generates imgaes based on the prompt and 5 different styles and then with additional guidence of hue loss</h4>")
    gr.HTML("<h6 align = 'center'> !!The image generation may take 5 to 10 minutes on CPU!!</h4>")

    with gr.Row():
        content = gr.Textbox(label = "Enter prompt text here")
        gr.Examples([
            "A mouse",
            "A puppy"
        ],
        inputs = content)
        num_steps = gr.Slider(1, 50, step = 1, value=30, label="Number of inference steps", info="Choose between 1 and 50")
        # gr.Number(value = 10, label = "Number of inference steps")

    
    with gr.Row():
        stl_dropdown = gr.Dropdown(
            stl_list, 
            value=stl_list[:1], multiselect=True, label="Style", 
            info="Styles to be applied on images"
        )
        size_dropdown = gr.Dropdown(
            [*img_size_opt_dict], 
            value = [*img_size_opt_dict][-1],
            label="Image size", info="Target size for generated images"
        )
    
    inputs = [
                content,
                num_steps,
                stl_dropdown,
                size_dropdown
                ]
    
    generate_btn = gr.Button(value = 'Generate')

    with gr.Row():
        with gr.Column(scale=2):
            wo_add_guide = gr.Gallery(
            label="Without additional guidence", show_label=True, elem_id="gallery"
            , columns=[3], rows=[2], object_fit="contain", height="auto")

        with gr.Column(scale=2):
            add_guide = gr.Gallery(
            label="With hue loss guidence", show_label=True, elem_id="gallery"
            , columns=[3], rows=[2], object_fit="contain", height="auto")
        outputs  = [wo_add_guide, add_guide ]
    generate_btn.click(fn = generate_images, inputs= inputs, outputs = outputs)

# for collab
demo.launch(debug=True)

# if __name__ == '__main__':
#     demo.launch()
