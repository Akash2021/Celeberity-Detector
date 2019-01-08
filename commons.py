import io
from torchvision import models
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torch
def get_model():
    checkpoint_path='classifier79.pt'
    model=models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024,512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Linear(512,5),
                                 nn.LogSoftmax(dim=1)
                                )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),strict=False)
    model.eval()
    return model

def get_tensor(image_bytes) :
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image=Image.open(io.BytesIO(image_bytes))
    return test_transforms(image).unsqueeze(0)
