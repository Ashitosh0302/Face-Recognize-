import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# ==================== PART 1: LOAD DATASET ====================
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

# Set dataset path
dataset_path = "dataset/faces"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset folder not found at: {dataset_path}")
    print("Please create folder structure: dataset/faces/person_name/images")
    print("Creating sample structure...")
    os.makedirs("dataset/faces/person1", exist_ok=True)
    os.makedirs("dataset/faces/person2", exist_ok=True)
    print("Created: dataset/faces/person1/")
    print("Created: dataset/faces/person2/")
    print("Please add face images to these folders and run again.")
    exit()

# Load images
X = []  # Face images as vectors
y = []  # Labels (person IDs)
class_names = []  # Person names
h, w = 100, 100  # Resize all images to 100x100

person_id = 0
print("\nLoading faces from dataset...")

# Go through each person's folder
for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):
        print(f"  Loading: {person_name}")
        class_names.append(person_name)
        
        # Load all images for this person
        image_count = 0
        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_folder, img_file)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize
                    resized = cv2.resize(gray, (w, h))
                    
                    # Normalize pixel values to [0, 1]
                    normalized = resized / 255.0
                    
                    # Flatten to vector
                    X.append(normalized.flatten())
                    y.append(person_id)
                    image_count += 1
        
        if image_count > 0:
            person_id += 1
        else:
            print(f"    Warning: No images found for {person_name}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\nDataset loaded successfully!")
print(f"Total images: {len(X)}")
print(f"Number of persons: {len(class_names)}")
print(f"Persons: {class_names}")

# ==================== PART 2: SPLIT DATA ====================
print("\n" + "=" * 60)
print("STEP 2: SPLITTING DATA (60% train, 40% test)")
print("=" * 60)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ==================== PART 3: PCA (EIGENFACES) ====================
print("\n" + "=" * 60)
print("STEP 3: APPLYING PCA (EIGENFACES METHOD)")
print("=" * 60)

# Apply PCA with 150 components (you can change this)
n_components = min(150, X_train.shape[0], X_train.shape[1])
num_eigenfaces_to_show = min(12, n_components)
print(f"Extracting {n_components} eigenfaces...")

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Training data after PCA: {X_train_pca.shape}")
print(f"Test data after PCA: {X_test_pca.shape}")
print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

# Show eigenfaces
print("\nDisplaying first 12 eigenfaces...")
eigenfaces = pca.components_.reshape((n_components, h, w))

num_eigenfaces_to_show = min(12, n_components)

plt.figure(figsize=(12, 8))
for i in range(num_eigenfaces_to_show):
    plt.subplot(3, 4, i + 1)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.title(f'Eigenface {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/eigenfaces.png', dpi=100, bbox_inches='tight')
plt.show()


# ==================== PART 4: TRAIN ANN ====================
print("\n" + "=" * 60)
print("STEP 4: TRAINING ANN (ARTIFICIAL NEURAL NETWORK)")
print("=" * 60)

print("Training ANN with backpropagation...")
ann = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 -> 50
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=True
)

ann.fit(X_train_pca, y_train)

# Test accuracy
train_score = ann.score(X_train_pca, y_train)
test_score = ann.score(X_test_pca, y_test)

print(f"\nANN Training Complete!")
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# ==================== PART 5: EXPERIMENT WITH K VALUES ====================
print("\n" + "=" * 60)
print("STEP 5: EXPERIMENT WITH DIFFERENT K VALUES")
print("=" * 60)

# Test different numbers of PCA components
max_components = min(X_train.shape[0], X_train.shape[1])
# Only keep k-values <= max_components
k_values = [k for k in [10, 30, 50, 80, 100, 120, 150] if k <= max_components]

if len(k_values) == 0:
    # If no valid k-values, just use all possible components
    k_values = list(range(1, max_components + 1))

print(f"Using PCA components: {k_values}")
accuracies = []

print("\nTesting different k values (PCA components):")
for k in k_values:
    # Apply PCA with k components
    pca_k = PCA(n_components=k, svd_solver='randomized', whiten=True)
    X_train_k = pca_k.fit_transform(X_train)
    X_test_k = pca_k.transform(X_test)
    
    # Train ANN
    ann_k = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        max_iter=300,
        random_state=42,
        verbose=False
    )
    ann_k.fit(X_train_k, y_train)
    
    # Calculate accuracy
    accuracy = ann_k.score(X_test_k, y_test)
    accuracies.append(accuracy)
    print(f"  k = {k:3d}: Accuracy = {accuracy:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of PCA Components (k)')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Number of PCA Components')
plt.grid(True, alpha=0.3)

# Add value labels on points
for i, (k, acc) in enumerate(zip(k_values, accuracies)):
    plt.annotate(f'{acc:.3f}', (k, acc), xytext=(0, 10),
                textcoords='offset points', ha='center')

plt.tight_layout()
plt.savefig('results/accuracy_vs_k.png', dpi=100, bbox_inches='tight')
plt.show()

# Find best k
best_idx = np.argmax(accuracies)
best_k = k_values[best_idx]
print(f"\nBest k value: {best_k} with accuracy: {accuracies[best_idx]:.4f}")

# ==================== PART 6: IMPOSTOR TESTING ====================
print("\n" + "=" * 60)
print("STEP 6: IMPOSTOR TESTING")
print("=" * 60)

# Create impostor faces (people not in training set)
print("Creating impostor faces...")
n_impostors = 20
impostor_data = []

for i in range(n_impostors):
    # Take a real face and add noise to create impostor
    idx = np.random.randint(0, X_test.shape[0])
    face = X_test[idx].copy()
    
    # Add random noise
    noise = np.random.normal(0, 0.2, face.shape)
    impostor_face = np.clip(face + noise, 0, 1)
    impostor_data.append(impostor_face)

impostor_data = np.array(impostor_data)

# Transform impostors using PCA
impostor_pca = pca.transform(impostor_data)

# Get predictions for impostors
impostor_probs = ann.predict_proba(impostor_pca)
impostor_confidences = np.max(impostor_probs, axis=1)

# Set confidence threshold
threshold = 0.7
rejected = np.sum(impostor_confidences < threshold)
impostor_accuracy = rejected / n_impostors

print(f"Number of impostors: {n_impostors}")
print(f"Confidence threshold: {threshold}")
print(f"Impostors correctly rejected: {rejected}/{n_impostors}")
print(f"Impostor rejection rate: {impostor_accuracy:.2%}")

# Plot impostor confidence scores
plt.figure(figsize=(10, 6))
plt.hist(impostor_confidences, bins=10, alpha=0.7, color='red', edgecolor='black')
plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
plt.xlabel('Prediction Confidence')
plt.ylabel('Number of Impostors')
plt.title('Impostor Test Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/impostor_test.png', dpi=100, bbox_inches='tight')
plt.show()

# ==================== PART 7: VISUALIZE RESULTS ====================
print("\n" + "=" * 60)
print("STEP 7: VISUALIZING PREDICTIONS")
print("=" * 60)

# Make predictions on test set
y_pred = ann.predict(X_test_pca)
y_prob = ann.predict_proba(X_test_pca)

# Display sample predictions
print("\nSample predictions from test set:")
plt.figure(figsize=(15, 10))

# Select 12 random test samples
n_samples = min(12, len(X_test))
indices = np.random.choice(len(X_test), n_samples, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(3, 4, i + 1)
    
    # Display face
    face_img = X_test[idx].reshape(h, w)
    plt.imshow(face_img, cmap='gray')
    
    # Get prediction info
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = np.max(y_prob[idx])
    
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    
    # Set title with color coding
    if true_label == pred_label:
        color = 'green'
        result = "✓ CORRECT"
    else:
        color = 'red'
        result = "✗ WRONG"
    
    plt.title(f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}\n{result}", 
              color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/test_predictions.png', dpi=100, bbox_inches='tight')
plt.show()

# ==================== PART 8: FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("1. DATASET")
print(f"   • Total images: {len(X)}")
print(f"   • Persons: {len(class_names)}")
print(f"   • Persons: {class_names}")

print("\n2. PCA RESULTS")
print(f"   • Components used: {n_components}")
print(f"   • Variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

print("\n3. ANN PERFORMANCE")
print(f"   • Test accuracy: {test_score:.4f}")
print(f"   • Best k value: {best_k}")
print(f"   • Best accuracy: {accuracies[best_idx]:.4f}")

print("\n4. IMPOSTOR TEST")
print(f"   • Impostors tested: {n_impostors}")
print(f"   • Rejection rate: {impostor_accuracy:.2%}")

print("\n5. OUTPUTS SAVED")
print("   • results/eigenfaces.png")
print("   • results/accuracy_vs_k.png")
print("   • results/impostor_test.png")
print("   • results/test_predictions.png")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)