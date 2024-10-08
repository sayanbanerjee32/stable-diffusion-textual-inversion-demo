# Stable diffusion textual inversion demo

## Objective

1. select 5 different styles from "community-created SD concepts library" and show output for the same prompt using these 5 different styles.
2. implement a variant of additional guidance loss and generate images using the same prompts used above to show differences. An example of such loss is `blue_loss` - when applied the generated images will be saturated with blue colour.
3. Convert this to HuggingFace Spaces app.

## Steps

### Experiment

Used this [github page](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) and created the [notebook](https://github.com/sayanbanerjee32/stable-diffusion-textual-inversion-demo/blob/main/SD_textual_inversion_guidence.ipynb) for experiment.

Below are the 5 SD concept library styles selected and example of generated image for the prompt "A puppy"

1. [birb-style](https://huggingface.co/sd-concepts-library/birb-style)  
  ![image](https://github.com/user-attachments/assets/a94594b1-aca3-4baf-9bc9-3874711e5923)

3. [cute-game-style](https://huggingface.co/sd-concepts-library/cute-game-style)  
   ![image](https://github.com/user-attachments/assets/42187c70-97cd-4465-bd5e-f7fa7b3dd1bc)
   
5. [depthmap](https://huggingface.co/sd-concepts-library/depthmap)  
   ![image](https://github.com/user-attachments/assets/b4b870f9-8ff0-4a81-a104-3a5695864f1a)

6. [line-art](https://huggingface.co/sd-concepts-library/line-art)  
   ![image](https://github.com/user-attachments/assets/927a2a89-076b-43b4-8183-45489f098d1a)

8. [low-poly-hd-logos-icons](https://huggingface.co/sd-concepts-library/low-poly-hd-logos-icons)  
    ![image](https://github.com/user-attachments/assets/775f8674-3f13-4f6a-92c0-4b4e9a780f98)

Experimented with couple of different guidance losses.

#### Hue Loss
Line Blue loss Hue loss will increase the color saturation of the images. See few examples of generated images below.
1. [birb-style](https://huggingface.co/sd-concepts-library/birb-style)  
  ![image](https://github.com/user-attachments/assets/446b9f8f-4816-4049-8f78-b5e91ac74852)

3. [cute-game-style](https://huggingface.co/sd-concepts-library/cute-game-style)  
   ![image](https://github.com/user-attachments/assets/aacb9723-b98a-4322-8a3a-8daf22fab13f)

4. [depthmap](https://huggingface.co/sd-concepts-library/depthmap)  
   ![image](https://github.com/user-attachments/assets/d3e56f9c-87bb-411f-baba-c57c4f9cf7c2)

6. [line-art](https://huggingface.co/sd-concepts-library/line-art)  
   ![image](https://github.com/user-attachments/assets/ec34cf51-df5c-462a-9126-3d81b4012094)

8. [low-poly-hd-logos-icons](https://huggingface.co/sd-concepts-library/low-poly-hd-logos-icons)  
  ![image](https://github.com/user-attachments/assets/7c316a54-1771-44f3-953c-fdbac8742641)


#### Image-text similarity loss
The idea for this loss was to allow for another additional prompt that can control some aspect of image generation. Such example prompts can be - white background, blurred, low quality etc.

See few examples of generated images with additional guidance prompt of "Low quality" below.

1. [birb-style](https://huggingface.co/sd-concepts-library/birb-style)  
  ![image](https://github.com/user-attachments/assets/6129ce26-791d-4839-86e0-60a7f5b49119)

2. [cute-game-style](https://huggingface.co/sd-concepts-library/cute-game-style)  
   ![image](https://github.com/user-attachments/assets/ab83ac23-0e7a-45c0-b452-155c28161d59)

4. [depthmap](https://huggingface.co/sd-concepts-library/depthmap)  
   ![image](https://github.com/user-attachments/assets/d53f1276-985c-4d1c-a895-0ca55755cd04)

6. [line-art](https://huggingface.co/sd-concepts-library/line-art)  
   ![image](https://github.com/user-attachments/assets/542f26bb-b6a3-49e8-b446-acf538311765)

8. [low-poly-hd-logos-icons](https://huggingface.co/sd-concepts-library/low-poly-hd-logos-icons)  
  ![image](https://github.com/user-attachments/assets/a987e338-64de-4296-b2bd-31390817015d)

## The HuggingFace Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/sayanbanerjee32/stable-diffusion-textual-inversion-demo)

Example with Hue loss  
![image](https://github.com/user-attachments/assets/baf27ccd-fc00-4431-8899-7bc320c89539)

Example with image-text similarity with additional prompt of "White background"  
![image](https://github.com/user-attachments/assets/2e29b5cd-b7d5-4e56-a28d-8ea8b144e5c0)


- The app takes a prompt and number of inference steps as input 
- It provides a dropdown for 5 of the styles mentioned. This is a multi-select dropdown.
  1. [birb-style](https://huggingface.co/sd-concepts-library/birb-style)  
  2. [cute-game-style](https://huggingface.co/sd-concepts-library/cute-game-style)  
  3. [depthmap](https://huggingface.co/sd-concepts-library/depthmap)  
  4. [line-art](https://huggingface.co/sd-concepts-library/line-art)  
  5. [low-poly-hd-logos-icons](https://huggingface.co/sd-concepts-library/low-poly-hd-logos-icons)  

- The app provides an option of target size for generated images. There are 3 options
  1. 512x512 - best quality but very slow
  2. 256x256 - not good quality but still slow 
  3. 128x128 - poor quality but faster
- The app provides option to select the loss to be applied - Hue loss or Text-Image similarity loss. If Text-Image Similarity loss is selected, then the app takes the additional guidance prompt as input as well.

## Challenges
1. While implementing Hue loss was straight forward, implementation of image-text similarity loss was not easy due to the loss calculation is dependent on the encoding of image and additional guidance text using CLIP model (pre-trained). 
2. The inference timing on CPU is very high. Different options were explored without much of success. Using a lower image size for generated image is some what useful but that compromises the quality to a great extent. Having a GPU for inferencing can solve this problem, but that comes with a cost.


