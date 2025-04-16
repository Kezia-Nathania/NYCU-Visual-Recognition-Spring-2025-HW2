import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import csv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom Dataset Loader (COCO Format)
class DigitDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(DigitDataset, self).__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super(DigitDataset, self).__getitem__(idx)
        boxes = [[x, y, x + w, y + h] for x, y, w, h in (obj['bbox']
                                                         for obj in target)]
        labels = [obj['category_id'] - 1 for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = target[0]['image_id'] if 'image_id' in target[0] else idx
        target = {'boxes': boxes, 'labels': labels,
                  'image_id': torch.tensor([image_id])}
        if self.transforms:
            img = self.transforms(img)
        return img, target


def calculate_map(model, data_loader, device, threshold=0.5):
    model.eval()
    coco_gt = data_loader.dataset.coco
    coco_dt = []
    num_image = sum(1 for _ in data_loader)
    preds = ['-1' for i in range(0, num_image)]

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [transform(image).to(device) for image in images]
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy().tolist()
                scores = output['scores'].cpu().numpy().tolist()
                labels = output['labels'].cpu().numpy().tolist()
                image_id = targets[i]['image_id'].item()

                for box, score, label in sorted(zip(boxes, scores, labels),
                                                key=lambda x: x[0]):
                    if score > threshold:
                        coco_dt.append({
                            'image_id': image_id,
                            'category_id': label + 1,
                            'bbox': [box[0], box[1], box[2] - box[0],
                                     box[3] - box[1]],
                            'score': score
                        })
                        results.append({
                            'image_id': image_id,
                            'bbox': [box[0], box[1], box[2] - box[0],
                                     box[3] - box[1]],
                            'score': score,
                            'category_id': label + 1
                        })
                        if preds[image_id - 1] == '-1':
                            preds[image_id - 1] = str(label)
                        else:
                            preds[image_id - 1] += str(label)

    coco_dt = coco_gt.loadRes(coco_dt)
    coco_evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    with open("pred.json", "w") as file:
        json.dump(results, file, indent=4)

    # Extract and print mAP from the summary statistics
    map_value = coco_evaluator.stats[0]
    print(f'Mean Average Precision (mAP): {map_value:.4f}')

    # Define the output file name
    output_file = 'pred.csv'

    # Write to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['image_id', 'pred_label'])

        # Write the data
        writer.writerows(zip(range(1, num_image + 1), preds))

    print(f'CSV file "{output_file}" has been created with {len(preds)} rows.')


class DigitTestDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_files = sorted([
            file for file in os.listdir(root)
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        image_id = os.path.splitext(self.image_files[idx])[0]
        return img, image_id

    def __len__(self):
        return len(self.image_files)


def test(model, data_loader, device, threshold=0.65):
    model.eval()

    # Initialize results container
    coco_dt = []
    results = []
    num_image = len(data_loader.dataset)
    print(f"Number of test images: {num_image}")

    preds = ['-1' for _ in range(num_image)]  # Placeholder for predictions

    with torch.no_grad():
        for idx, (images, image_ids) in tqdm(enumerate(data_loader),
                                             desc='Evaluating'):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']
                image_id = image_ids[i]

                # Apply NMS
                keep_idxs = nms(boxes, scores, iou_threshold=0.12)
                boxes = boxes[keep_idxs]
                scores = scores[keep_idxs]
                labels = labels[keep_idxs]

                boxes = boxes.cpu().numpy().tolist()
                scores = scores.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                for box, score, label in sorted(zip(boxes, scores, labels),
                                                key=lambda x: x[0]):
                    if score > threshold:
                        # Create the COCO-style result for each detection
                        coco_dt.append({
                            'image_id': image_id,
                            'category_id': label + 1,
                            'bbox': [box[0], box[1], box[2] - box[0],
                                     box[3] - box[1]],
                            'score': score
                        })
                        results.append({
                            'image_id': int(image_id),
                            'bbox': [box[0], box[1], box[2] - box[0],
                                     box[3] - box[1]],
                            'score': score,
                            'category_id': label + 1
                        })

                        # Update predictions (used for CSV output)
                        if preds[int(image_id) - 1] == '-1':
                            preds[int(image_id) - 1] = str(label)
                        else:
                            preds[int(image_id) - 1] += str(label)

    # Save predictions to pred.json
    with open("pred.json", "w") as file:
        json.dump(results, file, indent=4)
    print('json file "pred.json" has been created.')

    # Create a CSV for predictions (image_id, pred_label)
    output_file = 'pred.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'pred_label'])
        writer.writerows(zip(range(1, num_image + 1), preds))

    print(f'CSV file "{output_file}" has been created with {len(preds)} rows.')


# Transformations with data augmentation
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load dataset
train_dataset = DigitDataset('nycu-hw2-data/train', 'nycu-hw2-data/train.json')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          collate_fn=lambda x: tuple(zip(*x)))

# Custom sizes and aspect ratios
custom_sizes = ((64,), (96,), (128,), (256,), (384,))
custom_aspect_ratios = ((0.5, 1.0, 1.5),) * len(custom_sizes)

anchor_generator = AnchorGenerator(
    sizes=custom_sizes,
    aspect_ratios=custom_aspect_ratios
)

# Use pretrained ResNet101 with FPN
backbone = resnet_fpn_backbone('resnet101', pretrained=True)
num_classes = 11  # 10 classes + background
model = FasterRCNN(backbone, num_classes=num_classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.rpn.anchor_generator = anchor_generator
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.\
    FastRCNNPredictor(in_features, num_classes)

# Unfreeze Backbone
for param in model.backbone.parameters():
    param.requires_grad = True

model.to(device)

# Optimizer and Scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Variables for tracking loss
train_losses = []

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    for images, targets in pbar:
        images = [transform(image).to(device) for image in images]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Accumulate loss and update progress
        epoch_loss += losses.item()
        pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})

    # Store the average loss for this epoch
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'model101_epoch_{epoch+1}.pth')

    # Step the learning rate scheduler
    scheduler.step()

# Plotting the training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o',
         color='royalblue', linewidth=2)
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.xticks(range(1, epochs + 1))
plt.tight_layout()

# Save the styled plot as an image file
plt.savefig('learning_curve.png')

plt.close()
print('Training complete!')

results = []


val_dataset = DigitDataset('nycu-hw2-data/valid', 'nycu-hw2-data/valid.json')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda x: tuple(zip(*x)))

# Calculate mAP after training
calculate_map(model, val_loader, device)

results = []
test_dataset = DigitTestDataset('nycu-hw2-data/test', transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calculate mAP after training
test(model, test_loader, device)
