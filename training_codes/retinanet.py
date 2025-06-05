import os
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm import tqdm

CLASSES = ['__background__', 'Bus', 'Bike', 'Car', 'Pedestrian', "Truck"]  # 0 arka plan

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label in CLASSES:
            labels.append(CLASSES.index(label))
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            
           
            if xmin >= xmax or ymin >= ymax:
                print(f"Uyarı: Geçersiz bbox bulundu: {xmin}, {ymin}, {xmax}, {ymax}. Atlanıyor.")
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
        else:
            print(f"Uyarı: {label} etiketi CLASSES listesinde bulunamadı. Atlanıyor.")
    
    return boxes, labels

class VOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.ids = [f.split(".")[0] for f in os.listdir(ann_dir) if f.endswith(".xml")]
        print(f"Toplam {len(self.ids)} görüntü bulundu.")
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        img_path = os.path.join(self.img_dir, img_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_id + ".jpg")
            
        xml_path = os.path.join(self.ann_dir, img_id + ".xml")
        
        img = Image.open(img_path).convert("RGB")
        boxes, labels = parse_voc_xml(xml_path)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target
    
    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    with tqdm(data_loader, desc=f"Epoch {epoch+1}") as pbar:
        for images, targets in pbar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
           
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
        
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
           
            total_loss += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} - Ortalama kayıp: {avg_loss:.4f}")
    return avg_loss

def main():
    data_root = "data"  # Bu yolu kendi veri setinize göre değiştirin
    train_img_dir = os.path.join(data_root, "data/train/images")
    train_ann_dir = os.path.join(data_root, "data/train/annotations")
    
    # Test verileri için yollar
    test_img_dir = os.path.join(data_root, "data/test/images")
    test_ann_dir = os.path.join(data_root, "data/test/annotations")
    
    # Dönüşümler
    transform = T.Compose([
        T.ToTensor(),
        T.ColorJitter(),
        T.Normalize(),
        T.RandomAffine(),
        T.RandomRotation()
    ])
    
    train_dataset = VOCDataset(train_img_dir, train_ann_dir, transforms=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetHead
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    
    num_classes = len(CLASSES)
    
    backbone = model.backbone
    
    anchor_generator = model.anchor_generator
    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )
    
    
    model.head = head
   
    model.to(device)
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
   
    num_epochs = 50
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        lr_scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/best_retinanet_model.pth")
            print(f"En iyi model kaydedildi: best_retinanet_model.pth (Kayıp: {best_loss:.4f})")
        
        torch.save(model.state_dict(), f"models/retinanet_model_epoch_{epoch+1}.pth")
    
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main()