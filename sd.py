import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os

from hue_loss import hue_loss
from text_image_similarity_loss import text_image_similarity_loss

from functools import partial


torch.manual_seed(1)
# if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

from huggingface_hub import hf_hub_download


stl_list = [
            'birb-style',
            'cute-game-style',
            'depthmap',
            'line-art',
            'low-poly-hd-logos-icons'
            ]

for stl in stl_list:
    if not os.path.exists(stl):
        os.mkdir(stl)
    hf_hub_download(repo_id=f"sd-concepts-library/{stl}", filename="learned_embeds.bin", local_dir=f"./{stl}")

img_size_opt_dict = {
    "512x512 - best quality but very slow": (512,512), 
    "256x256 - not good quality but still slow" :  (256,256), 
    "128x128 - poor quality  but faster" : (128,128),
    }

loss_fn_dict = {
    'Hue Loss': hue_loss,
    'Text-Image Similarity Loss': text_image_similarity_loss,
}
# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);

# Convert latents to images

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

#Generating an image with these modified embeddings
def generate_with_embs(text_embeddings, text_input, loss_fn = None, loss_scale = 200, guidance_scale = 7.5,
                       seed_value = 1, num_inference_steps = 50, additional_guidence = False, hight_width = (512, 512)):
    height, width = hight_width                        # default height of Stable Diffusion
    # width = 512                         # default width of Stable Diffusion
    # num_inference_steps = 50            # Number of denoising steps
                   # Scale for classifier-free guidance
    generator = torch.manual_seed(seed_value)   # Seed generator to create the inital latent noise
    batch_size = 1


    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #### ADDITIONAL GUIDANCE ###
        if i%5 == 0 and additional_guidence:
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()

            # Get the predicted x0:
            latents_x0 = latents - sigma * noise_pred
            # latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

            # Decode to image space
            denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

            # Calculate loss
            loss = loss_fn(denoised_images) * loss_scale

            # Occasionally print it out
            if i%10==0:
                print(i, 'loss:', loss.item())

            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]

            # Modify the latents based on this gradient
            # latents = latents.detach() - cond_grad * sigma**2
            latents = latents.detach() - cond_grad * sigma**2

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Ensure the latents do not lose the grad tracking
        # latents.requires_grad_()

    return latents_to_pil(latents)[0]

def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output

# Access the embedding layer
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
pos_emb_layer = text_encoder.text_model.embeddings.position_embedding

position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)


def generate_images(prompt, num_inference_steps, stl_list, img_size, loss_fn_option, text_loss_text = None):
    ### add a statis text that will contain the style
    prompt = prompt + ' in the style of puppy'
    height_width = img_size_opt_dict[img_size]
    loss_fn = loss_fn_dict[loss_fn_option]
    # Tokenize
    text_input = tokenizer(prompt, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)

    # Get token embeddings
    token_embeddings = token_emb_layer(input_ids)

    wo_guide_lst = []
    guide_lst = []
    for i, stl in enumerate(stl_list):
        stl_embed = torch.load(f'{stl}/learned_embeds.bin')

        # The new embedding - our special birb word
        replacement_token_embedding = stl_embed[f'<{stl}>'].to(torch_device)

        # Insert this into the token embeddings
        token_embeddings[0, min(torch.where(input_ids[0]==tokenizer.eos_token_id)[0]) - 1] = replacement_token_embedding.to(torch_device)

        # Combine with pos embs
        input_embeddings = token_embeddings + position_embeddings

        #  Feed through to get final output embs
        modified_output_embeddings = get_output_embeds(input_embeddings)

        # # And generate an image with this:
        pil_im = generate_with_embs(modified_output_embeddings,
                                    num_inference_steps = num_inference_steps,
                                    text_input = text_input,
                                    seed_value = i,additional_guidence = False,
                                    hight_width = height_width)
        wo_guide_lst.append((pil_im,stl))

        # defaults
        loss_scale = 200
        guidance_scale = 7.5

        if loss_fn == text_image_similarity_loss and text_loss_text is not None:
            loss_fn = partial(text_image_similarity_loss, target_text = text_loss_text)
            loss_scale = 50
            guidance_scale = 20

        pil_im = generate_with_embs(modified_output_embeddings,
                                    num_inference_steps = num_inference_steps,
                                    text_input = text_input,
                                    loss_fn = loss_fn,
                                    additional_guidence = True,
                                    hight_width = height_width,
                                    loss_scale = loss_scale,
                                    guidance_scale = guidance_scale,
                                    seed_value = i)
        guide_lst.append((pil_im,stl))

    return wo_guide_lst, guide_lst
