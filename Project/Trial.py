import torch
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

model = models.resnet50(pretrained=True)

image_dir = 'guns.v3i.coco'
annotation_file = 'guns.v3i.coco/train/_annotations.coco.json'

transform = transforms.Compose([
    transforms.ToTensor(),
])

coco_dataset = CocoDetection(root=image_dir, annFile=annotation_file, transform=transform)

data_loader = DataLoader(coco_dataset, batch_size=16, shuffle=True, num_workers=4)

for images, targets in data_loader:
    # images: tensor of shape [batch_size, C, H, W]
    # targets: list of dictionaries with bounding boxes and class labels
    print(targets)  # Print annotations for each image

    

