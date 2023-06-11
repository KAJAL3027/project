import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision

import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)  # adjust weight size
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define the prediction function
def predict_image(image_path, model):
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')
    # Apply the transformation pipeline
    image_tensor = transform(image)
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
    # Get the predicted class index
    predicted_index = torch.argmax(output).item()
    prob = torch.nn.functional.softmax(output, dim=1)
    return predicted_index, prob[0][predicted_index].item()

if __name__ == "__main__":
    model_path = 'model_90pct_acc_v1.pt'
    model = torch.load(model_path)
    model.eval()
    result = predict_image(r'C:\Users\pranj\OneDrive\project\static\uploads\093.png', model)
    print(result)