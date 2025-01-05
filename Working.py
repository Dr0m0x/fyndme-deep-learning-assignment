import torch
import torch.nn as nn
import torch.nn.functional as F

class FyndMeModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FyndMeModel, self).__init__()

        # Image branch: CNN for image embeddings
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Text branch: FCN for text embeddings
        self.text_fc = nn.Sequential(
            nn.Linear(10, 64),  # 10-dimensional input (e.g., one-hot encoded digits)
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, image, text):
        # Process image and text inputs
        image_embedding = self.image_cnn(image)
        text_embedding = self.text_fc(text)

        # Normalize embeddings for cosine similarity
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return image_embedding, text_embedding

    def compute_similarity(self, image_embedding, text_embedding):
        # Cosine similarity between embeddings
        similarity = torch.sum(image_embedding * text_embedding, dim=1)
        return similarity

# Example usage
if __name__ == "__main__":
    # Create the model
    model = FyndMeModel()

    # Example inputs
    dummy_image = torch.randn(8, 1, 28, 28)  # Batch of 8 MNIST-like images
    dummy_text = torch.randn(8, 10)  # Batch of 8 one-hot encoded labels

    # Forward pass
    image_embedding, text_embedding = model(dummy_image, dummy_text)

    # Compute similarity
    similarity = model.compute_similarity(image_embedding, text_embedding)

    print("Image Embeddings Shape:", image_embedding.shape)
    print("Text Embeddings Shape:", text_embedding.shape)
    print("Similarity Scores:", similarity)
