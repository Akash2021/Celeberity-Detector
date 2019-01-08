from commons import get_tensor, get_model
import torchvision
model = get_model()
def get_flower_name(image_bytes) :
    tensor=get_tensor(image_bytes)
    outputs=model.forward(tensor)
    print(outputs)
    a,prediction = outputs.max(1)
    category=prediction.item()
    trainset=torchvision.datasets.ImageFolder('train/')
    category=trainset.classes[prediction.item()]
    return category
