import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
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

    return predicted_index

if __name__ == "__main__":
    model_path = 'model_90pct_acc_v1.pt'
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict_image(r'C:\Users\pranj\OneDrive\project\static\uploads\093.png', model)