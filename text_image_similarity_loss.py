
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(torch_device)
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_text_embedding(text):
    text_tokens = clip_tokenizer([text]).to(torch_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_image_embedding(image):
    image_input = clip_preprocess(image).unsqueeze(0).to(torch_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def text_image_similarity_loss(generated_images, target_text = "plain background"):
    # Get text embedding
    text_embedding = get_text_embedding(target_text)

    # Ensure the generated_images have requires_grad=True
    # generated_images.requires_grad_(True)

    # Convert image tensor to the required format (normalization, resizing)
    # Normalize the images (assuming they are in [0, 1])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example size, modify as needed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformation
    transformed_images = transform(generated_images)

    # Assuming `image_encoder` is a pretrained model that returns image embeddings
    # Get image embeddings
    # image_embeddings = image_encoder(generated_images)
    with torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(transformed_images).float()
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)


    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(norm_image_features, text_embedding, dim=-1)

    # Define the loss as 1 - cosine similarity (assuming we want to maximize similarity)
    loss = 1 - cos_sim.mean()

    return loss
