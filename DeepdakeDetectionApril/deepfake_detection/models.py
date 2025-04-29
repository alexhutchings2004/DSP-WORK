import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from collections import OrderedDict

#=====================
# Model Utilities
#=====================

def get_model_size(model):
    """Calculate model size in MB - helps with deployment planning."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_model(model_name, num_classes=2, pretrained=True, feature_extraction=False):
    """
    Instantiate a model based on name with appropriate options.
    
    Args:
        model_name: Architecture name (resnet18, efficientnet, etc.)
        num_classes: Number of output classes (2 for deepfake detection)
        pretrained: Whether to use pretrained weights
        feature_extraction: Whether to use feature extraction capabilities
    
    Returns:
        model: Instantiated PyTorch model
    """
    if model_name == 'resnet18':
        if feature_extraction:
            model = FeatureExtractorResNet(num_classes=num_classes, pretrained=pretrained)
        else:
            model = models.resnet18(weights='DEFAULT' if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'efficientnet':
        try:
            if feature_extraction:
                model = FeatureExtractorEfficientNet(num_classes=num_classes, pretrained=pretrained)
            else:
                model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        except Exception as e:
            print(f"Error loading EfficientNet: {e}")
            print("Trying alternative EfficientNet implementation...")
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model

#=====================
# Custom CNN Model
#=====================

class CustomDeepfakeDetector(nn.Module):
    """Custom CNN architecture designed specifically for deepfake detection."""
    def __init__(self, num_classes=2):
        super(CustomDeepfakeDetector, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#=====================
# Advanced Deepfake Detector
#=====================

class AdvancedDeepfakeDetector(nn.Module):
    """Advanced CNN with feature extraction capabilities for better analysis."""
    def __init__(self, num_classes=2):
        super(AdvancedDeepfakeDetector, self).__init__()
        
        # Feature extraction layers with named stages for better feature extraction
        self.stage1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        
        self.stage2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        
        self.stage3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        
        self.stage4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU()),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature layers
        self.feature_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction through different stages
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Apply attention mechanism
        att = self.attention(x4)
        x4_att = x4 * att
        
        # Extract features
        features = self.feature_layers(x4_att)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x):
        """
        Extract features from different layers for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            dict: Dictionary with features from different layers
        """
        features = {}
        
        x1 = self.stage1(x)
        features['stage1'] = x1
        
        x2 = self.stage2(x1)
        features['stage2'] = x2
        
        x3 = self.stage3(x2)
        features['stage3'] = x3
        
        x4 = self.stage4(x3)
        features['stage4'] = x4
        
        att = self.attention(x4)
        features['attention'] = att
        
        x4_att = x4 * att
        features['attended'] = x4_att
        
        x_feat = self.feature_layers(x4_att)
        features['embedding'] = x_feat
        
        logits = self.classifier(x_feat)
        features['logits'] = logits
        
        return features
    
    def get_attention_map(self, x):
        """
        Get the attention map for an input image to see what regions are important.
        
        Args:
            x: Input tensor
            
        Returns:
            tensor: Attention map
        """
        # Forward pass up to the attention mechanism
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Get attention map
        att = self.attention(x4)
        
        return att

#=====================
# ResNet Feature Extractor
#=====================

class FeatureExtractorResNet(nn.Module):
    """ResNet18 with feature extraction capabilities for deepfake detection."""
    def __init__(self, num_classes=2, pretrained=True):
        super(FeatureExtractorResNet, self).__init__()
        
        # Load base model
        weights = 'DEFAULT' if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
        self.edge_layer = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.noise_layer = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.artifact_layer = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Base ResNet feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Apply attention
        att = self.attention(x4)
        x_att = x4 * att
        
        # Global average pooling
        x_pool = self.avgpool(x_att)
        x_flat = torch.flatten(x_pool, 1)
        
        # Classification
        x_out = self.fc(x_flat)
        
        return x_out
    
    def extract_features(self, x):
        """
        Extract features from different layers for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            dict: Dictionary with features from different layers
        """
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features['stem'] = x
        
        x1 = self.layer1(x)
        features['layer1'] = x1
        
        # Extract edge features
        edge_features = self.edge_layer(x1)
        features['edge_features'] = edge_features
        
        x2 = self.layer2(x1)
        features['layer2'] = x2
        
        x3 = self.layer3(x2)
        features['layer3'] = x3
        
        # Extract noise features
        noise_features = self.noise_layer(x3)
        features['noise_features'] = noise_features
        
        x4 = self.layer4(x3)
        features['layer4'] = x4
        
        # Extract artifact features
        artifact_features = self.artifact_layer(x4)
        features['artifact_features'] = artifact_features
        
        # Attention mechanism
        att = self.attention(x4)
        features['attention'] = att
        
        x_att = x4 * att
        features['attended'] = x_att
        
        x_pool = self.avgpool(x_att)
        x_flat = torch.flatten(x_pool, 1)
        features['embedding'] = x_flat
        
        x_out = self.fc(x_flat)
        features['logits'] = x_out
        
        return features

#=====================
# EfficientNet Feature Extractor
#=====================

class FeatureExtractorEfficientNet(nn.Module):
    """EfficientNet with feature extraction capabilities for deepfake detection."""
    def __init__(self, num_classes=2, pretrained=True):
        super(FeatureExtractorEfficientNet, self).__init__()
        
        # Load base model
        weights = 'DEFAULT' if pretrained else None
        try:
            self.base_model = models.efficientnet_b0(weights=weights)
        except Exception as e:
            print(f"Error loading EfficientNet: {e}")
            print("Using alternative implementation...")
            from torchvision.models import EfficientNet_B0_Weights
            w = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.base_model = models.efficientnet_b0(weights=w)
        
        # Extract feature layers from base model for better access

        self.features = self.base_model.features
        
        self.texture_layer = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.quality_layer = nn.Conv2d(112, 64, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        num_ftrs = self.base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        # Extract intermediate features for analysis
        features_dict = {}
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            
            # Save specific layers for analysis
            if idx == 3:  # Early layer
                features_dict['early'] = x
            elif idx == 5:  # Quality-relevant layer
                features_dict['quality'] = x
                quality_features = self.quality_layer(x)
                features_dict['quality_processed'] = quality_features
            elif idx == 7:  # Texture-relevant layer
                features_dict['texture'] = x
                texture_features = self.texture_layer(x)
                features_dict['texture_processed'] = texture_features
        
        # Final features
        features = x
        
        att = self.attention(features)
        att_features = features * att
        
        # Global average pooling and classification
        avg_pooled = self.base_model.avgpool(att_features)
        avg_pooled = torch.flatten(avg_pooled, 1)
        output = self.classifier(avg_pooled)
        
        return output
    
    def extract_features(self, x):
        """
        Extract features from different layers for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            dict: Dictionary with features from different layers
        """
        features = {}
        
        # Process through feature extractor
        for idx, layer in enumerate(self.features):
            x = layer(x)
            
            # Save specific layers for analysis
            features[f'layer_{idx}'] = x
            
            if idx == 3:  # Early layer
                features['early'] = x
            elif idx == 5:  # Quality-relevant layer
                features['quality'] = x
                quality_features = self.quality_layer(x)
                features['quality_processed'] = quality_features
            elif idx == 7:  # Texture-relevant layer
                features['texture'] = x
                texture_features = self.texture_layer(x)
                features['texture_processed'] = texture_features
        
        # Final features
        final_features = x
        features['final'] = final_features
        
        # Apply attention
        att = self.attention(final_features)
        features['attention'] = att
        att_features = final_features * att
        features['attended'] = att_features
        
        # Global average pooling
        avg_pooled = self.base_model.avgpool(att_features)
        avg_pooled = torch.flatten(avg_pooled, 1)
        features['embedding'] = avg_pooled
        
        # Classification
        output = self.classifier(avg_pooled)
        features['logits'] = output
        
        return features

#=====================
# MobileNet Feature Extractor
#=====================

class FeatureExtractorMobileNet(nn.Module):
    """MobileNet with feature extraction capabilities for deepfake detection."""
    def __init__(self, num_classes=2, pretrained=True):
        super(FeatureExtractorMobileNet, self).__init__()
        
        # Load base model
        weights = 'DEFAULT' if pretrained else None
        try:
            self.base_model = models.mobilenet_v2(weights=weights)
        except Exception as e:
            print(f"Error loading MobileNet: {e}")
            print("Using alternative implementation...")
            from torchvision.models import MobileNet_V2_Weights
            w = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.base_model = models.mobilenet_v2(weights=w)
        
        # Extract feature extractor
        self.features = self.base_model.features
        
        # Add new feature extraction layers specific to deepfake detection
        self.compression_layer = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.artifact_layer = nn.Conv2d(160, 32, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        num_ftrs = self.base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        # Process through feature extractor
        for idx, layer in enumerate(self.features):
            x = layer(x)
        
        # Apply attention
        att = self.attention(x)
        x_att = x * att
        
        # Global average pooling
        x_pooled = nn.functional.adaptive_avg_pool2d(x_att, (1, 1))
        x_flat = torch.flatten(x_pooled, 1)
        
        # Classification
        x_out = self.classifier(x_flat)
        
        return x_out
    
    def extract_features(self, x):
        """
        Extract features from different layers for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            dict: Dictionary with features from different layers
        """
        features = {}
        
        # Process through feature extractor
        for idx, layer in enumerate(self.features):
            x = layer(x)
            features[f'layer_{idx}'] = x
            
            # Extract specific features for analysis
            if idx == 7:  # Compression artifacts
                comp_features = self.compression_layer(x)
                features['compression_features'] = comp_features
            elif idx == 14:  # Manipulation artifacts
                art_features = self.artifact_layer(x)
                features['artifact_features'] = art_features
        
        # Final features
        features['final'] = x
        
        # Apply attention
        att = self.attention(x)
        features['attention'] = att
        x_att = x * att
        features['attended'] = x_att
        
        # Global average pooling
        x_pooled = nn.functional.adaptive_avg_pool2d(x_att, (1, 1))
        x_flat = torch.flatten(x_pooled, 1)
        features['embedding'] = x_flat
        
        # Classification
        x_out = self.classifier(x_flat)
        features['logits'] = x_out
        
        return features

#=====================
# Multi-modal Detector
#=====================

class MultiModalDeepfakeDetector(nn.Module):
    """Multi-modal deepfake detector that processes different image representations."""
    def __init__(self, num_classes=2, modalities=['rgb', 'noise', 'freq']):
        super(MultiModalDeepfakeDetector, self).__init__()
        
        self.modalities = modalities
        self.modality_encoders = nn.ModuleDict()
        
        # Create encoder for each modality
        for modality in modalities:
            if modality == 'rgb':
                # RGB encoder - using ResNet18 backbone with reduced final layer
                rgb_model = models.resnet18(weights='DEFAULT')
                self.modality_encoders['rgb'] = nn.Sequential(
                    *list(rgb_model.children())[:-1],
                    nn.Flatten(),
                    nn.Linear(512, 256),
                    nn.ReLU()
                )
            elif modality == 'noise':
                # Noise pattern encoder - lighter model
                self.modality_encoders['noise'] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
            elif modality in ['freq', 'dct']:
                # Frequency domain encoder
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
            elif modality == 'edges':
                # Edge features encoder
                self.modality_encoders['edges'] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 64),
                    nn.ReLU()
                )
        
        # Calculate total feature size
        total_features = 0
        for modality in modalities:
            if modality == 'rgb':
                total_features += 256
            elif modality == 'noise':
                total_features += 128
            elif modality in ['freq', 'dct']:
                total_features += 128
            elif modality == 'edges':
                total_features += 64
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(128, num_classes)
        
        # Attention weights for modalities
        self.modality_attention = nn.Parameter(torch.ones(len(modalities), requires_grad=True))
    
    def forward(self, x):
        """
        Forward pass - expects a dictionary of tensors, one for each modality.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            output: Model output logits
        """
        # Process each modality
        features = []
        
        for i, modality in enumerate(self.modalities):
            if modality in x:
                # Forward through the encoder
                feat = self.modality_encoders[modality](x[modality])
                
                # Apply modality attention
                att_weight = torch.sigmoid(self.modality_attention[i])
                feat = feat * att_weight
                
                features.append(feat)
        
        # Concatenate features
        if len(features) > 0:
            fused_features = torch.cat(features, dim=1)
            
            # Fusion layer
            fused = self.fusion(fused_features)
            
            # Classification
            output = self.classifier(fused)
            
            return output
        else:
            # Fallback if no features were provided
            device = next(self.parameters()).device
            return torch.zeros((x[list(x.keys())[0]].shape[0], 2), device=device)
    
    def extract_modality_importance(self):
        """
        Extract the importance of each modality based on attention weights.
        
        Returns:
            dict: Dictionary with modality importance scores
        """
        att_weights = torch.sigmoid(self.modality_attention).detach().cpu().numpy()
        
        # Create a dictionary of modality importance
        importance = {}
        for i, modality in enumerate(self.modalities):
            importance[modality] = float(att_weights[i])
        
        return importance

#=====================
# Dual-stream Detector
#=====================

class DualStreamDeepfakeDetector(nn.Module):
    """Dual-stream deepfake detector that processes RGB and noise patterns in parallel."""
    def __init__(self, num_classes=2):
        super(DualStreamDeepfakeDetector, self).__init__()
        
        # RGB stream - for visual content
        rgb_model = models.resnet18(weights='DEFAULT')
        self.rgb_encoder = nn.Sequential(
            *list(rgb_model.children())[:-1],
            nn.Flatten()
        )
        
        # Noise stream - optimized for detecting noise patterns and artifacts
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Stream attention weights
        self.rgb_attention = nn.Parameter(torch.tensor(1.0))
        self.noise_attention = nn.Parameter(torch.tensor(1.0))
    
    def extract_noise_pattern(self, image_tensor):
        """
        Extract noise pattern from image tensor.
        
        Args:
            image_tensor: Input tensor [B, C, H, W]
            
        Returns:
            noise_tensor: Tensor with noise pattern
        """
        # Convert to numpy, process, and convert back to tensor
        batch_size = image_tensor.size(0)
        device = image_tensor.device
        noise_tensors = []
        
        for i in range(batch_size):
            img = image_tensor[i].cpu().permute(1, 2, 0).numpy()
            
            # Apply gaussian blur
            blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            
            # Subtract the blurred image from the original to get the noise
            noise = img - blur
            
            # Convert back to tensor
            noise_tensor = torch.from_device(noise.transpose(2, 0, 1), device=device)
            noise_tensors.append(noise_tensor)
        
        # Stack back into a batch
        return torch.stack(noise_tensors)
    
    def forward(self, x):
        # Process the RGB stream
        rgb_features = self.rgb_encoder(x)
        rgb_att = torch.sigmoid(self.rgb_attention)
        rgb_features = rgb_features * rgb_att
        
        # Extract and process noise patterns
        noise_tensor = self.extract_noise_pattern(x)
        noise_features = self.noise_encoder(noise_tensor)
        noise_att = torch.sigmoid(self.noise_attention)
        noise_features = noise_features * noise_att
        
        # Concatenate features from both streams
        combined = torch.cat((rgb_features, noise_features), dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output
    
    def stream_importance(self):
        """
        Get the importance of each stream based on attention weights.
        
        Returns:
            tuple: (rgb_importance, noise_importance)
        """
        rgb_importance = torch.sigmoid(self.rgb_attention).item()
        noise_importance = torch.sigmoid(self.noise_attention).item()
        
        return rgb_importance, noise_importance