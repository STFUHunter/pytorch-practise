from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from torchvision import transforms
import torch
import torch.nn as nn
from mnist import CNN

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

app = FastAPI()
@app.post('/predict/')
async def predict(file : UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        prob = torch.nn.functional.softmax(outputs, dim=-1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_class].item()
    
    return {
        "predicted_digit": pred_class,
        "confidence": round(confidence, 4)
    }