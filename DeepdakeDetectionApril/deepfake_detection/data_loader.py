import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
import albumentations as A
from facenet_pytorch import MTCNN
import random

#=====================
# custom transforms
#=====================

class AddGaussianNoise:
    """add gaussian noise to tensor - helps with model robustness"""
    def __init__(self, mean=0., std=0.03):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

#=====================
# standard dataset loader
#=====================

class DeepfakeDataset(Dataset):
    """dataset for loading real and fake face images with preprocessing options"""
    
    def __init__(self, root_dir, transform=None, face_detection=False, noise_analysis=False, 
                 erase_aug=False, freq_analysis=False):
        """
        args:
            root_dir: directory containing real and fake folders
            transform: optional transforms to apply
            face_detection: whether to crop faces before processing
            noise_analysis: whether to extract noise patterns
            erase_aug: whether to use random erasing for augmentation
            freq_analysis: whether to include frequency domain features
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Real', 'Fake']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # initialize face detector if needed
        self.face_detection = face_detection
        if face_detection:
            self.face_detector = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
        # set other preprocessing flags
        self.noise_analysis = noise_analysis
        self.erase_aug = erase_aug
        self.freq_analysis = freq_analysis
        
        # random erasing transform - helps model learn from partial faces
        self.random_erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        
        # collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def detect_faces(self, image):
        """detect and crop faces to focus on the important part"""
        try:
            # convert to numpy for opencv/mtcnn
            img_np = np.array(image)
            # detect faces
            boxes, _ = self.face_detector.detect(img_np)
            
            if boxes is not None and len(boxes) > 0:
                # get the first face with a margin
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                
                # add margin around face for context
                h, w = img_np.shape[:2]
                margin = int(min(w, h) * 0.1)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                # crop the face
                face = img_np[y1:y2, x1:x2]
                return Image.fromarray(face)
            else:
                return image
        except Exception as e:
            print(f"face detection error: {e}")
            return image
    
    def extract_noise_pattern(self, image):
        """
        extract noise patterns using high-pass filter - deepfakes usually have 
        unique noise signatures that this helps expose
        """
        # convert to numpy array
        img_np = np.array(image).astype(np.float32)
        
        # apply gaussian blur
        blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
        
        # subtract the blurred image to get high-frequency components
        noise = img_np - blur
        
        # normalize to visible range
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # convert back to PIL image
        noise_img = Image.fromarray(noise)
        
        return noise_img
    
    def apply_frequency_analysis(self, image):
        """
        transform image to frequency domain - manipulations often leave 
        artifacts in frequency space that aren't visible in pixel space
        """
        # convert to grayscale and numpy array
        gray = np.array(image.convert('L'))
        
        # apply discrete fourier transform
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # compute magnitude spectrum (log for better visualization)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        
        # normalize and convert back to PIL
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        freq_img = Image.fromarray(magnitude_spectrum)
        
        # resize to match original image size and convert to RGB
        freq_img = freq_img.resize(image.size)
        return freq_img.convert('RGB')
    
    def __getitem__(self, idx):
        """get a single sample - handles all preprocessing and transforms"""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # apply face detection if enabled
        if self.face_detection:
            image = self.detect_faces(image)
            
        # apply transforms to the image
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # default transformation if none provided
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
            
        # apply random erasing if enabled (for training)
        if self.erase_aug:
            image_tensor = self.random_erasing(image_tensor)
        
        # for noise analysis, return both original and noise pattern
        if self.noise_analysis:
            noise_image = self.extract_noise_pattern(image)
            if self.transform:
                noise_tensor = self.transform(noise_image)
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                noise_tensor = transform(noise_image)
            return (image_tensor, noise_tensor), label
        
        # for frequency analysis, return both spatial and frequency domain
        if self.freq_analysis:
            freq_image = self.apply_frequency_analysis(image)
            if self.transform:
                freq_tensor = self.transform(freq_image)
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                freq_tensor = transform(freq_image)
            return (image_tensor, freq_tensor), label
            
        return image_tensor, label

#=====================
# advanced data augmentation
#=====================

def get_advanced_transformations(img_size=224):
    """
    create robust augmentation pipelines specialized for deepfake detection
    
    args:
        img_size: target image size
    
    returns:
        train_transform: transformations for training
        val_test_transform: transformations for validation/testing
    """
    # training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # simulate different image qualities with blur
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # add noise to simulate different image qualities
        transforms.RandomApply([
            AddGaussianNoise(mean=0., std=0.03)
        ], p=0.3),
    ])
    
    # validation and testing transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

#=====================
# dataloader creation
#=====================

def get_data_loaders(data_dir, batch_size=32, img_size=224, seed=42, 
                     face_detection=False, noise_analysis=False, 
                     freq_analysis=False):
    """
    create train, validation and test dataloaders with appropriate transforms
    
    args:
        data_dir: base directory with train/valid/test folders
        batch_size: batch size for training
        img_size: target image size
        seed: for reproducibility
        face_detection: whether to use face detection
        noise_analysis: whether to extract noise patterns
        freq_analysis: whether to use frequency domain features
        
    returns:
        train_loader, val_loader, test_loader: dataloaders for each split
    """
    # set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # get transforms for each split
    train_transform, val_test_transform = get_advanced_transformations(img_size)
    
    # create datasets for each split (using existing folder structure)
    train_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_dir, 'train'), 
        transform=train_transform,
        face_detection=face_detection,
        noise_analysis=noise_analysis,
        erase_aug=True,  # only apply random erasing to training
        freq_analysis=freq_analysis
    )
    
    val_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_dir, 'valid'), 
        transform=val_test_transform,
        face_detection=face_detection,
        noise_analysis=noise_analysis,
        erase_aug=False,
        freq_analysis=freq_analysis
    )
    
    test_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_dir, 'test'), 
        transform=val_test_transform,
        face_detection=face_detection,
        noise_analysis=noise_analysis,
        erase_aug=False,
        freq_analysis=freq_analysis
    )
    
    # create data loaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

#=====================
# multimodal dataset
#=====================

class DeepfakeMultiModalDataset(Dataset):
    """advanced dataset for loading multiple image representations simultaneously"""
    
    def __init__(self, root_dir, transform=None, modalities=['rgb', 'noise', 'freq']):
        """
        args:
            root_dir: directory containing real and fake folders
            transform: optional transforms to apply
            modalities: which image representations to use
                        (rgb, noise, freq, dct, edges)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Real', 'Fake']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.modalities = modalities
        
        # initialize face detector
        self.face_detector = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def extract_noise_pattern(self, image):
        """extract noise pattern to catch manipulation artifacts"""
        img_np = np.array(image).astype(np.float32)
        blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
        noise = img_np - blur
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(noise)
    
    def extract_frequency_features(self, image):
        """extract frequency domain features to find inconsistencies"""
        gray = np.array(image.convert('L'))
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(magnitude_spectrum).convert('RGB')
    
    def extract_dct_features(self, image):
        """extract dct features which highlight compression artifacts"""
        gray = np.array(image.convert('L'))
        # apply discrete cosine transform
        dct = cv2.dct(np.float32(gray))
        # log scale for better visualization
        dct_log = np.log(np.abs(dct) + 1)
        # normalize to 0-255
        dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(dct_norm).convert('RGB')
    
    def extract_edge_features(self, image):
        """extract edge features which can reveal facial inconsistencies"""
        img_np = np.array(image)
        # convert to grayscale for edge detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        # apply canny edge detector
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges).convert('RGB')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """get multiple representations of the same image"""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # try to detect and crop face
        try:
            img_np = np.array(image)
            boxes, _ = self.face_detector.detect(img_np)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                h, w = img_np.shape[:2]
                margin = int(min(w, h) * 0.1)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                face = img_np[y1:y2, x1:x2]
                image = Image.fromarray(face)
        except Exception as e:
            print(f"face detection error: {e}")
        
        # process each modality
        modality_tensors = {}
        
        # basic transform if none provided
        basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if self.transform is None else self.transform
        
        # process each requested modality
        if 'rgb' in self.modalities:
            modality_tensors['rgb'] = basic_transform(image)
        
        if 'noise' in self.modalities:
            noise_image = self.extract_noise_pattern(image)
            modality_tensors['noise'] = basic_transform(noise_image)
        
        if 'freq' in self.modalities:
            freq_image = self.extract_frequency_features(image)
            modality_tensors['freq'] = basic_transform(freq_image)
        
        if 'dct' in self.modalities:
            dct_image = self.extract_dct_features(image)
            modality_tensors['dct'] = basic_transform(dct_image)
        
        if 'edges' in self.modalities:
            edge_image = self.extract_edge_features(image)
            modality_tensors['edges'] = basic_transform(edge_image)
        
        return modality_tensors, label

#=====================
# multimodal dataloader creation
#=====================

def get_multimodal_data_loaders(data_dir, batch_size=32, img_size=224, seed=42, modalities=['rgb', 'noise', 'freq']):
    """
    create train, validation and test dataloaders for multimodal training
    
    args:
        data_dir: base directory with train/valid/test folders
        batch_size: batch size for training
        img_size: target image size
        seed: for reproducibility
        modalities: which image representations to use
        
    returns:
        train_loader, val_loader, test_loader: dataloaders for each split
    """
    # set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # define common transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # create datasets for each split
    train_dataset = DeepfakeMultiModalDataset(
        root_dir=os.path.join(data_dir, 'train'), 
        transform=train_transform,
        modalities=modalities
    )
    
    val_dataset = DeepfakeMultiModalDataset(
        root_dir=os.path.join(data_dir, 'valid'), 
        transform=val_test_transform,
        modalities=modalities
    )
    
    test_dataset = DeepfakeMultiModalDataset(
        root_dir=os.path.join(data_dir, 'test'), 
        transform=val_test_transform,
        modalities=modalities
    )
    
    # create data loaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader