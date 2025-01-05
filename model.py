import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class FyndMeModel(nn.Module):
    def __init__(self, text_input_dim=10, embedding_dim=128):
        super(FyndMeModel, self).__init__()

        # Image branch: Convolutional Neural Network (CNN) with Batch Normalization
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Text branch: Fully connected network for text embeddings
        self.text_fc = nn.Sequential(
            nn.Linear(text_input_dim, 64),  # Configurable text input dimension
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Cosine similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, text):
        # Pass image through the CNN branch
        image_embedding = self.image_cnn(image)

        # Pass text through the text embedding branch
        text_embedding = self.text_fc(text)

        # Normalize embeddings to ensure similarity scores are cosine-based
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)

        return image_embedding, text_embedding

    def compute_similarity(self, image_embedding, text_embedding):
        # Compute cosine similarity between image and text embeddings
        return self.cosine_similarity(image_embedding, text_embedding)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity_scores, labels):
        # Labels: 1 for similar pairs, 0 for dissimilar pairs
        positive_loss = (1 - labels) * (similarity_scores ** 2)
        negative_loss = labels * (F.relu(self.margin - similarity_scores + 1e-6) ** 2)
        loss = torch.mean(positive_loss + negative_loss)
        return loss

class MNISTTextDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]

        # Convert label to one-hot encoding
        text_label = torch.zeros(10)
        text_label[label] = 1.0

        return image, text_label

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        images, text_labels = batch
        images, text_labels = images.to(device), text_labels.to(device)

        # Forward pass
        image_embeddings, text_embeddings = model(images, text_labels)
        similarity_scores = model.compute_similarity(image_embeddings, text_embeddings)

        # Compute loss
        labels = torch.ones_like(similarity_scores).to(device)  # Assuming all pairs are similar
        loss = criterion(similarity_scores, labels)

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            images, text_labels = batch
            images, text_labels = images.to(device), text_labels.to(device)

            # Forward pass
            image_embeddings, text_embeddings = model(images, text_labels)
            similarity_scores = model.compute_similarity(image_embeddings, text_embeddings)

            # Compute loss
            labels = torch.ones_like(similarity_scores).to(device)  # Assuming all pairs are similar
            loss = criterion(similarity_scores, labels)
            total_loss += loss.item()

            total_similarity += similarity_scores.sum().item()
            count += len(similarity_scores)

    avg_similarity = total_similarity / count
    avg_loss = total_loss / len(dataloader)
    return avg_similarity, avg_loss

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    embedding_dim = 128
    learning_rate = 0.001
    num_epochs = 10

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset with data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_dataset = MNISTTextDataset(mnist_train)
    test_dataset = MNISTTextDataset(mnist_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = FyndMeModel(embedding_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        avg_similarity, val_loss = evaluate_model(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Avg Similarity: {avg_similarity:.4f}")

    # Final Evaluation
    avg_similarity, test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Average Similarity Score on Test Set: {avg_similarity:.4f}")

    # Save model
    torch.save(model.state_dict(), "fyndme_model.pth")
    print("Model saved to 'fyndme_model.pth'.")
