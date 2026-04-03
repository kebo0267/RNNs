# imports
import gc
import gradio as gr
import torch
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Predefine some image resolutions for the dropdown menu
img_resolutions = [{"(328x720) Banner Ad": (328, 720)},
                   {"(720x328) Poster": (720, 328)},
                   {"(512x512) Square": (512, 512)}]

# Clear out the GPU memory before starting
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# Take model and generate an image from prompt, negative prompt, guidance scale, and seed
def generate_image (prompt, negative_prompt, guidance_scale, resolution, seed):
    # get height and width from img_resolutions based on resolution selected in dropdown
    h, w = [res[resolution] for res in img_resolutions if list(res.keys())[0] == resolution][0]
    if device != "cuda":
        torchType = None        
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torchType, use_safetensors=True, variant="fp16")
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    generator = torch.Generator(device).manual_seed(seed)
    pipe.to(device)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    image = pipe(prompt, negative_prompt=negative_prompt, generator=generator, height=h, width=w, guidance_scale=guidance_scale).images
    
    # Clear out the GPU memory before starting a new image
    # Needed on my machine as it would fill GPU VRAM and then crash after a few runs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return image[0]

# Set a default prompt for the textbox input
def_prompt = "Create a banner ad for 'Gears Ltd.' They make gears. Use blue for background color, and steel for the gears"
# Launch the Gradio interface with the defined inputs and outputs
demo = gr.Interface(fn=generate_image , inputs=[gr.Textbox(label="Prompt", value=def_prompt,
                                                        info="Describe the image you want to generate. Be specific about colors, styles, and composition."),
                                                gr.Textbox(label="Negative Prompt", value="text, words, logo, watermarks, signatures",
                                                        info="Describe elements you want to avoid in the generated image."),
                                                gr.Slider(0, 20, step=0.1, label="Guidance Scale", value=7.5,
                                                        info="Higher values make the image more closely follow the prompt, but can reduce creativity. Adjust to find the right balance."),
                                                gr.Dropdown(label="Resolution", choices=[list(res.keys())[0] for res in img_resolutions], value=list(img_resolutions[0].keys())[0],
                                                        info="Select the resolution for the generated image."),
                                                gr.Number(label="Seed", value=691,
                                                        info="Set a seed for reproducibility.")],
                                        outputs=[gr.Image(label="Banner Background")])

demo.launch()