import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import time
import json
from tqdm.notebook import tqdm
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms as T
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_JSON = os.path.join(TRAIN_DIR, "train.json")
TEST_JSON = os.path.join(TEST_DIR, "test.json")
TRAIN_IMAGES = os.path.join(TRAIN_DIR, "images")  
TEST_IMAGES = os.path.join(TEST_DIR, "images") 

print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TRAIN_JSON: {TRAIN_JSON}")
print(f"TRAIN_IMAGES: {TRAIN_IMAGES}")
print(f"TEST_IMAGES: {TEST_IMAGES}")

print(f"TRAIN_DIR mevcut mu: {os.path.exists(TRAIN_DIR)}")
print(f"TRAIN_JSON mevcut mu: {os.path.exists(TRAIN_JSON)}")
print(f"TRAIN_IMAGES mevcut mu: {os.path.exists(TRAIN_IMAGES)}")
print(f"TEST_IMAGES mevcut mu: {os.path.exists(TEST_IMAGES)}")



class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = image.shape[-1] if isinstance(image, torch.Tensor) else image.width
            
            image = F.hflip(image)
            
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image, target):
        orig_width = image.shape[-1] if isinstance(image, torch.Tensor) else image.width
        orig_height = image.shape[-2] if isinstance(image, torch.Tensor) else image.height
        
        image = F.resize(image, self.size)
        
        scale_width = self.size[1] / orig_width
        scale_height = self.size[0] / orig_height
        
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] *= scale_width   # x_min
            boxes[:, 1] *= scale_height  # y_min
            boxes[:, 2] *= scale_width   # x_max
            boxes[:, 3] *= scale_height  # y_max
            target["boxes"] = boxes
            
        return image, target


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.color_jitter = T.ColorJitter(brightness=brightness, 
                                         contrast=contrast, 
                                         saturation=saturation, 
                                         hue=hue)
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if not isinstance(image, torch.Tensor):
                image = self.color_jitter(image)
            else:
              
                pil_image = F.to_pil_image(image)
                pil_image = self.color_jitter(pil_image)
                image = F.to_tensor(pil_image)
        return image, target


class RandomRotation:
    def __init__(self, degrees, prob=0.5, expand=False):
        self.degrees = degrees
        self.prob = prob
        self.expand = expand
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Convert tensor to PIL if necessary
            is_tensor = isinstance(image, torch.Tensor)
            if is_tensor:
                pil_image = F.to_pil_image(image)
            else:
                pil_image = image
            
            width, height = pil_image.size
            center = (width / 2, height / 2)
            
            rotated_image = F.rotate(pil_image, angle, expand=self.expand)
            
         
            if is_tensor:
                image = F.to_tensor(rotated_image)
            else:
                image = rotated_image
                
            new_width, new_height = rotated_image.size
            
            # Rotate boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                
                
                center_boxes = torch.zeros_like(boxes)
                center_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
                center_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
                center_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]        # w
                center_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]        # h
                
               
                angle_rad = -angle * np.pi / 180  
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)
                
                new_x = (center_boxes[:, 0] - center[0]) * cos_theta - \
                        (center_boxes[:, 1] - center[1]) * sin_theta + center[0]
                new_y = (center_boxes[:, 0] - center[0]) * sin_theta + \
                        (center_boxes[:, 1] - center[1]) * cos_theta + center[1]
                
                # Adjust for expand=True
                if self.expand:
                   
                    scale_x = new_width / width
                    scale_y = new_height / height
                    
                    
                    new_x = new_x * scale_x
                    new_y = new_y * scale_y
                    center_boxes[:, 2] = center_boxes[:, 2] * scale_x
                    center_boxes[:, 3] = center_boxes[:, 3] * scale_y
                
                
                center_boxes[:, 0] = new_x
                center_boxes[:, 1] = new_y
                
                # Convert back to [x1, y1, x2, y2] format
                boxes[:, 0] = center_boxes[:, 0] - center_boxes[:, 2] / 2
                boxes[:, 1] = center_boxes[:, 1] - center_boxes[:, 3] / 2
                boxes[:, 2] = center_boxes[:, 0] + center_boxes[:, 2] / 2
                boxes[:, 3] = center_boxes[:, 1] + center_boxes[:, 3] / 2
                
                final_width = image.shape[-1] if is_tensor else image.width
                final_height = image.shape[-2] if is_tensor else image.height
                
                boxes[:, 0].clamp_(0, final_width)
                boxes[:, 1].clamp_(0, final_height)
                boxes[:, 2].clamp_(0, final_width)
                boxes[:, 3].clamp_(0, final_height)
                
                # Filter out invalid boxes
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                
                # Update target
                target["boxes"] = boxes[keep]
                if "labels" in target:
                    target["labels"] = target["labels"][keep]
                
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height = image.shape[-2] if isinstance(image, torch.Tensor) else image.height
            
            image = F.vflip(image)
            
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class RandomResizedCrop:
    def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33), prob=0.5):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            is_tensor = isinstance(image, torch.Tensor)
            if is_tensor:
                pil_image = F.to_pil_image(image)
            else:
                pil_image = image
                
            width, height = pil_image.size
            
            area = width * height
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            w = min(w, width)
            h = min(h, height)
            
            # Get random crop position
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            
            # Perform crop and resize
            cropped_image = F.crop(pil_image, i, j, h, w)
            resized_image = F.resize(cropped_image, self.size)
            
            if is_tensor:
                image = F.to_tensor(resized_image)
            else:
                image = resized_image
            
            # Update bounding boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                
                boxes[:, 0] = boxes[:, 0] - j  # x_min
                boxes[:, 1] = boxes[:, 1] - i  # y_min
                boxes[:, 2] = boxes[:, 2] - j  # x_max
                boxes[:, 3] = boxes[:, 3] - i  # y_max
                
                scale_w = self.size[1] / w
                scale_h = self.size[0] / h
                
                # Apply scaling
                boxes[:, 0] = boxes[:, 0] * scale_w
                boxes[:, 1] = boxes[:, 1] * scale_h
                boxes[:, 2] = boxes[:, 2] * scale_w
                boxes[:, 3] = boxes[:, 3] * scale_h
                
                # Clip boxes to image boundaries
                boxes[:, 0].clamp_(0, self.size[1])
                boxes[:, 1].clamp_(0, self.size[0])
                boxes[:, 2].clamp_(0, self.size[1])
                boxes[:, 3].clamp_(0, self.size[0])
                
                # Filter out invalid boxes
                keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                
                # Update target
                target["boxes"] = boxes[keep]
                if "labels" in target:
                    target["labels"] = target["labels"][keep]
                
        return image, target


class SimpleTransforms:
    def __init__(self, train=True, input_size=None):
        transforms_list = []
        
        transforms_list.append(ToTensor())
        
        if train:
            transforms_list.append(RandomHorizontalFlip(prob=0.5))
        
        if input_size is not None:
            transforms_list.append(Resize(input_size))
        
        transforms_list.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomVerticalFlip(),
            RandomResizedCrop(),
            RandomHorizontalFlip()
        )
        
        self.transforms = Compose(transforms_list)
    
    def __call__(self, image, target):
        return self.transforms(image, target)


def list_image_samples(directory, limit=3):
    if os.path.exists(directory):
        files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            print(f"{directory} içindeki örnek görüntüler:")
            for f in files[:limit]:
                img_path = os.path.join(directory, f)
                try:
                    img = Image.open(img_path)
                    print(f"  - {f} ({img.size[0]}x{img.size[1]})")
                except Exception as e:
                    print(f"  - {f} (açılamadı: {e})")
        else:
            print(f"{directory} içinde görüntü dosyası bulunamadı!")
    else:
        print(f"{directory} dizini mevcut değil!")

list_image_samples(TRAIN_IMAGES)
list_image_samples(TEST_IMAGES)

def load_annotations(json_file):
    print(f"JSON dosyası yükleniyor: {json_file}")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"JSON dosyası başarıyla yüklendi. Veri türü: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Anahtarlar: {list(data.keys())[:5]} ...")
            
            if 'categories' in data:
                categories = {cat['id']: cat['name'] for cat in data['categories']}
                print(f"Kategoriler: {categories}")
            
            if 'images' in data:
                print(f"Görüntü sayısı: {len(data['images'])}")
                if data['images']:
                    print(f"İlk görüntü örneği: {data['images'][0]}")
            
            if 'annotations' in data:
                print(f"Annotation sayısı: {len(data['annotations'])}")
                if data['annotations']:
                    print(f"İlk annotation örneği: {data['annotations'][0]}")
                
        return data
    except Exception as e:
        print(f"JSON dosyası yüklenirken hata: {e}")
        return None

# COCO format dataset
class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.data = load_annotations(json_file)
        
        if self.data is None or 'images' not in self.data or 'annotations' not in self.data:
            print("Geçerli COCO veri seti bulunamadı!")
            self.images_info = {}
            self.image_ids = []
            self.annotations_by_image = {}
            self.categories = {}
            return
            
        self.images_info = {img['id']: img for img in self.data['images']}
        self.image_ids = list(self.images_info.keys())
        
        self.annotations_by_image = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}
        
        self.class_names = ['__background__']  
        for cat_id in sorted(self.categories.keys()):
            self.class_names.append(self.categories[cat_id])
        
        print(f"Veri seti yüklendi: {len(self.image_ids)} görüntü, {len(self.categories)} kategori")
        print(f"Kategoriler: {self.categories}")
        print(f"Sınıf adları (arka plan dahil): {self.class_names}")
        
        if self.image_ids:
            self._check_sample_image(self.image_ids[0])
    
    def _check_sample_image(self, img_id):
        """Örnek bir görüntüyü kontrol et"""
        try:
            image_info = self.images_info[img_id]
            img_filename = image_info['file_name']
            img_path = os.path.join(self.img_dir, img_filename)
            
            print(f"Örnek görüntü ID: {img_id}")
            print(f"Dosya adı: {img_filename}")
            print(f"Tam yol: {img_path}")
            print(f"Dosya var mı: {os.path.exists(img_path)}")
            
            if not os.path.exists(img_path):
                alt_path = os.path.join(self.img_dir, f"{img_id}.png")
                print(f"Alternatif yol deneniyor: {alt_path}")
                print(f"Alternatif dosya var mı: {os.path.exists(alt_path)}")
            
            ann_count = len(self.annotations_by_image.get(img_id, []))
            print(f"Bu görüntü için {ann_count} annotation var")
            
        except Exception as e:
            print(f"Örnek görüntü kontrolünde hata: {e}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        try:
            
            img_id = self.image_ids[idx]
            
          
            image_info = self.images_info[img_id]
            img_filename = image_info['file_name']
            
            
            img_path = os.path.join(self.img_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Hata: Görüntü dosyası bulunamadı: {img_path}")
                
                img = Image.new('RGB', (640, 480), color=(0, 0, 0))
            else:
                img = Image.open(img_path).convert("RGB")
            
            annotations = self.annotations_by_image.get(img_id, [])
            
            boxes = []
            labels = []
            
            for ann in annotations:
                try:
               
                    x, y, width, height = ann['bbox']
                    
                  
                    x_min = x
                    y_min = y
                    x_max = x + width
                    y_max = y + height
                    
                   
                    if width <= 0 or height <= 0:
                        continue
                   
                    category_id = ann['category_id'] + 1
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(category_id)
                    
                except Exception as e:
                    print(f"Annotation işlerken hata: {e}")
                    continue
            
           
            if len(boxes) == 0:
                boxes.append([0, 0, 10, 10])
                labels.append(0)  
            
           
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([img_id])
            
            if self.transforms is not None:
                img, target = self.transforms(img, target)
                
            return img, target
            
        except Exception as e:
            print(f"Veri yükleme hatası (idx={idx}): {e}")
         
            img = Image.new('RGB', (640, 480), color=(0, 0, 0))
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            target = {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
                "image_id": torch.tensor([0])
            }
            return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))


def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model(model, dataloader, optimizer, lr_scheduler, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} başlıyor")
        model.train()
        
        epoch_loss = 0.0
        running_loss = 0.0
        processed_batches = 0
        start_time = time.time()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in progress_bar:
            try:
               
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                
                optimizer.zero_grad()
                
              
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                
                losses.backward()
                optimizer.step()
                
                batch_loss = losses.item()
                running_loss += batch_loss
                epoch_loss += batch_loss
                processed_batches += 1
                
                if (batch_idx + 1) % 500 == 0:
                    avg_loss = running_loss / 10
                    elapsed_time = time.time() - start_time
                    examples_per_sec = 10 * len(images) / elapsed_time
                    print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}, "
                          f"Hız: {examples_per_sec:.1f} görüntü/sn")
                    running_loss = 0.0
                    start_time = time.time()
                    
            except Exception as e:
                print(f"Batch {batch_idx} işlenirken hata: {e}")
                continue
        
        lr_scheduler.step()
        
        if processed_batches > 0:
            avg_epoch_loss = epoch_loss / processed_batches
            print(f"Epoch {epoch+1} tamamlandı. Ortalama kayıp: {avg_epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} başarısız. Hiçbir batch başarıyla işlenemedi.")
        
        if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"models/faster_rcnn_epoch_{epoch+1}.pth")
            print(f"Model kaydedildi: models/faster_rcnn_epoch_{epoch+1}.pth")
    
    return model

def evaluate_model(model, dataloader, device):
            model.to(device)
            model.eval()
            
            results = []
            processed_images = 0
            
            with torch.no_grad():
                for images, targets in dataloader:
                    images = list(img.to(device) for img in images)
                    
                   
                    outputs = model(images)
                    
                   
                    for i, output in enumerate(outputs):
                        image_id = targets[i]["image_id"].item()
                        boxes = output["boxes"].cpu().numpy()
                        scores = output["scores"].cpu().numpy()
                        labels = output["labels"].cpu().numpy()
                        
                     
                        high_scores_idxs = np.where(scores > 0.5)[0]
                        processed_images += 1
                        
                        for idx in high_scores_idxs:
                            x1, y1, x2, y2 = boxes[idx]
                            
                          
                            pred_label = int(labels[idx])
                            if pred_label > 0:  
                                result = {
                                    "image_id": image_id,
                                    "category_id": pred_label - 1, 
                                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                    "score": float(scores[idx])
                                }
                                results.append(result)
                    
                    if processed_images % 100 == 0:
                        print(f"{processed_images} görüntü değerlendirildi, {len(results)} tahmin yapıldı")
            
            print(f"Toplam {processed_images} görüntü değerlendirildi, {len(results)} tahmin yapıldı")
            return results

def main():
    try:
        train_data = load_annotations(TRAIN_JSON)
        
        if train_data is None:
            print("Train verisi yüklenemedi, işlem durduruluyor.")
            return
            
        categories = {cat['id']: cat['name'] for cat in train_data['categories']}
        print(f"COCO kategori ID'leri: {list(categories.keys())}")
        print(f"COCO kategori adları: {list(categories.values())}")
        
        num_classes = max(categories.keys()) + 2
        print(f"Toplam sınıf sayısı (arka plan dahil): {num_classes}")
        
        print("\nTrain veri seti oluşturuluyor...")
        train_dataset = COCODataset(TRAIN_JSON, TRAIN_IMAGES, transforms=SimpleTransforms(train=True, input_size=800))
        
        print(f"Eğitim veri seti büyüklüğü: {len(train_dataset)}")
        
        batch_size = 6
        print(f"\nVeri yükleyici oluşturuluyor (batch_size={batch_size})...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  
            drop_last=False
        )
        
        print("\nModel oluşturuluyor...")
        model = get_faster_rcnn_model(num_classes)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
     
        print("\nModel eğitimi başlıyor...")
        num_epochs = 30
        model = train_model(model, train_loader, optimizer, lr_scheduler, device, num_epochs=num_epochs)
        
        
        torch.save(model.state_dict(), 'models/faster_rcnn_final.pth')
        print("\nFinal model kaydedildi: faster_rcnn_final.pth")
        
       
        print("\nTest veri seti oluşturuluyor...")
        test_dataset = COCODataset(TEST_JSON, TEST_IMAGES, transforms=SimpleTransforms(train=False, input_size=800))
        
       
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        
        print("\nModel değerlendirmesi başlıyor...")
        results = evaluate_model(model, test_loader, device)
        
        
        with open('predictions.json', 'w') as f:
            json.dump(results, f)
        print("\nTahminler kaydedildi: predictions.json")
        
    except Exception as e:
        print(f"\nAna kod yürütme hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()