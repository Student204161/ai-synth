import os
from src.models.model import MyImageClassifier
import torch
from PIL import Image
from torchvision import transforms

# Function to preprocess image for the NN
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)

    # Resize image while keeping aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    target_width, target_height = target_size
    resize_width = int(min(target_width, target_height * aspect_ratio))
    resize_height = int(min(target_height, target_width / aspect_ratio))
    resized_image = image.resize((resize_width, resize_height))

    # Create a black canvas with the target size
    canvas = Image.new("RGB", target_size, color="black")

    # Paste the resized image onto the canvas, centered
    x_offset = (target_width - resize_width) // 2
    y_offset = (target_height - resize_height) // 2
    canvas.paste(resized_image, (x_offset, y_offset))

    # Convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(canvas)
    return image_tensor


# Load the data
data_path = "/home/amapy/Documents/Spring_2024/ADLCV/Project/ai-synth_1/src/data/one_note_frames_classes_good/one_note_frames"
data_files = os.listdir(data_path)
train_data = {}
test_data = {}
validation_data = {}

for file in data_files:
    file_path = os.path.join(data_path, file)

    # testing data
    test_data[file] = str(os.path.join(data_path, file, file + "1.jpg"))
    
    #validation data
    validation_data[file] = str(os.path.join(data_path, file, "0.jpg"))

    # training and validation data
    for i in range(2, 6):
        train_data[file] = []
        train_data[file].append(str(os.path.join(data_path, file, file + str(i) + ".jpg")))

# Convert labels to integers
label_to_index = {label: index for index, label in enumerate(train_data.keys())}
index_to_label = {index: label for label, index in label_to_index.items()}

y_train = [label_to_index[label] for label in train_data.keys()]
y_test = [label_to_index[label] for label in test_data.keys()]

num_classes = len(y_train)
num_epochs = 5
batch_size = 4

# Instantiate the model
model = MyImageClassifier(in_channels=3, num_classes=num_classes)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for key, value in train_data.items():
        images = torch.stack([preprocess_image(image_path) for image_path in value])
        labels = torch.tensor([label_to_index[key]] * len(value))
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for key, value in test_data.items():
        images = torch.stack([preprocess_image(image_path) for image_path in value])
        labels = torch.tensor([label_to_index[key]] * len(value))
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images: %d %%' % (100 * correct / total))

# Map predicted integers back to labels
predicted_labels = [index_to_label[index.item()] for index in predicted]
print("Predicted labels:", predicted_labels)