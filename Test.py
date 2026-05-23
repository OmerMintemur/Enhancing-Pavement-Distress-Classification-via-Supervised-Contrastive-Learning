import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
import os
class CE_SupCon_Model(nn.Module):
    def __init__(self, backbone_name='resnet50', latent_dim=512, proj_dim=128, num_classes=3):
        super().__init__()
        if backbone_name == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in base.parameters():
                param.requires_grad = True
            self.encoder_cnn = nn.Sequential(*list(base.children())[:-2])
            feature_dim = 2048
        elif backbone_name == 'efficientnet_b0':
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            for param in base.parameters():
                param.requires_grad = True
            self.encoder_cnn = base.features
            feature_dim = 1280
        else:
            raise ValueError("Backbone not supported")

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(feature_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        feat = self.encoder_cnn(x)
        feat = self.avgpool(feat).flatten(1)
        h = self.encoder_mlp(feat)
        logits = self.classifier(h)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return logits, z, h

def evaluate_model(model, dataloader, device, which_set):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    temp_sum = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # print(f"{temp_sum}/{len(dataloader.dataset)}")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs,z,h = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            temp_sum = temp_sum + len(inputs)

    # Generate Report
    print(f"Accuracy on     {which_set} - {accuracy_score(all_labels, all_preds)}")
    print(f"F1 Score on     {which_set} - {f1_score(all_labels, all_preds, average='weighted')}")
    print(f"Precision on    {which_set} - {precision_score(all_labels, all_preds, average='weighted')}")
    print(f"Recall on       {which_set} - {recall_score(all_labels, all_preds, average='weighted')}")
MODEL_NAME = "resnet50"
model = None
data_dir = '..\\dataset'
model_folder = 'my_experiment_logsresnet50_Run_2'

model = CE_SupCon_Model(backbone_name=MODEL_NAME)

# Data Transforms and Model Preparation
input_size = 224
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


device = torch.device("cuda")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=128, shuffle=True) for x in ['train', 'val']}
state_dict = torch.load(f"{model_folder}\\best_model.pth", map_location=device)

print(MODEL_NAME)
model.load_state_dict(state_dict)
evaluate_model(model, dataloader=dataloaders['train'], device=device, which_set='train')
evaluate_model(model, dataloader=dataloaders['val'], device=device, which_set='val')