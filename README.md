# **FyndMe Deep Learning Assignment**

## **Project Overview**
This project involves developing a deep learning model that generates embeddings for images and text and computes a similarity score between matching pairs. The goal is to align visual and textual modalities effectively using the MNIST dataset and corresponding randomized text labels.

The task demonstrates the application of dual-branch neural networks and PyTorch's deep learning capabilities to process and align multimodal inputs.

---

## **Features**
- **Dual-Branch Model Architecture:**
  - A **Convolutional Neural Network (CNN)** branch for extracting features from images.
  - A **Fully Connected Network (FCN)** branch for processing one-hot encoded text labels.
- **Cosine Similarity Calculation:** 
  - Computes alignment between normalized embeddings for image-text pairs.
- **Custom Dataset Class:**
  - Includes randomized text labels to increase input diversity.
- **Contrastive Loss Function:**
  - Ensures embeddings for similar pairs are close and dissimilar pairs are far apart.

---

## **Dataset**
The MNIST dataset is used, containing grayscale images of handwritten digits (0–9). For each digit, a randomized textual representation is generated (e.g., "Three," "3," or "tree"). Text labels are provided using the helper function in `helpers.py`.

### **Preprocessing Steps:**
- **Image Normalization:** 
  - Pixel values are scaled to the range \([-1, 1]\).
- **Text Randomization:** 
  - Labels are converted into diverse textual forms using a mapping dictionary.

---

## **Model Architecture**
### **Image Branch (CNN):**
- Two convolutional layers with ReLU activation and batch normalization.
- Max-pooling layers for down-sampling.
- Fully connected layers to generate embeddings.

### **Text Branch (FCN):**
- Fully connected layers process one-hot encoded text labels to generate embeddings.

### **Similarity Metric:**
- **Cosine Similarity** measures alignment between the image and text embeddings.

### **Loss Function:**
- **Contrastive Loss** minimizes the distance between embeddings of similar pairs and penalizes dissimilar ones.

---

## **Project Structure**
project/ ├── data/ # Dataset directory ├── src/ # Source code │ ├── dataset.py # Custom dataset class │ ├── helpers.py # Text randomization and tokenizer │ ├── main.py # Training and validation script │ ├── model.py # Model and loss function ├── models/ # Saved trained models │ └── fyndme_model.pth # Trained model file ├── README.md # Project documentation ├── requirements.txt # Python dependencies


---

## **Training**
### **Setup:**
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 10

### **Training Process:**
1. Load MNIST images and randomized text labels.
2. Generate embeddings using CNN and FCN branches.
3. Compute similarity using cosine similarity.
4. Optimize embeddings using contrastive loss.

---

## **Evaluation**
### **Metrics:**
- **Validation Loss:** Monitored to ensure proper alignment of image-text pairs.
- **Similarity Score:** Measures the cosine similarity between embeddings.

---

## **Results**
- **High Alignment:** The model achieved high similarity scores (\(> 0.9\)) on the test dataset, indicating effective alignment of image and text embeddings.
- **Low Loss:** The contrastive loss converged to a very low value, validating the model's performance.

---

## **How to Run the Project**
### **Requirements:**
- Python 3.8+
- PyTorch
- torchvision
- transformers

### **Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/Dr0m0x/fyndme-deep-learning-assignment.git
   cd fyndme-deep-learning-assignment

Contributors
[Moriah/ Mo H./ Dr0m0]
Email: [Dr0m0py@gmail.com]
LinkedIn: [www.linkedin.com/in/dr0m0x]
