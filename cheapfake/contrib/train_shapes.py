import torch
from torchsummary import summary
import cheapfake.contrib.models_contrib as models

device = torch.device("cuda:0")

def face_encoder():
    model = models.FeaturesEncoder().to(device)
    summary(model, (2, 75, 68), batch_size=1)

def frame_encoder():
    model = models.FeaturesEncoder(models.NetworkType.lipnet).to(device)
    summary(model, (1, 75, 512))

def mmclassifier():
    model = models.MultimodalClassifier(device=device).to(device)
    summary(model, (3, 1, 256))

if __name__ == "__main__":
    #face_encoder()
    #frame_encoder()
    mmclassifier()
