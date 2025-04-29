import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import random
from time import time
import cv2
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.manifold import TSNE
from torch.autograd import Variable
import torchvision.transforms as transforms
from captum.attr import GradientShap, IntegratedGradients, Occlusion
from PIL import Image
import matplotlib.cm as cm

#=====================
# setup functions
#=====================

def set_seed(seed):
    """set random seeds for reproducibility - helps make experiments consistent"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#=====================
# visualization & reporting
#=====================

def plot_metrics(history, save_path=None):
    """
    plot training and validation metrics in a nice format
    
    args:
        history: training history dictionary with metrics
        save_path: where to save the plot (optional)
    """
    # create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # plot loss
    ax1.plot(history['train_loss'], label='training loss')
    ax1.plot(history['val_loss'], label='validation loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('training and validation loss')
    ax1.legend()
    ax1.grid(True)
    
    # plot accuracy
    ax2.plot(history['train_acc'], label='training accuracy')
    ax2.plot(history['val_acc'], label='validation accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_title('training and validation accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"metrics plot saved to {save_path}")
    
    plt.show()

def save_metrics_to_csv(history, filename):
    """
    save training metrics to a csv file for later analysis
    
    args:
        history: training history dictionary
        filename: output csv filename
    """
    df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'epoch_time': history['epoch_times']
    })
    
    df.to_csv(filename, index=False)
    print(f"metrics saved to {filename}")

def compare_models(results):
    """
    compare performance of different models with charts
    
    args:
        results: dictionary with model names as keys and results as values
    """
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    model_sizes = [results[name]['model_size'] for name in model_names]
    
    # create a dataframe for easy comparison
    comparison = pd.DataFrame({
        'Model': model_names,
        'Accuracy (%)': accuracies,
        'Training Time (s)': training_times,
        'Model Size (MB)': model_sizes
    })
    
    # sort by accuracy
    comparison = comparison.sort_values('Accuracy (%)', ascending=False)
    
    # print the comparison table
    print("\nmodel comparison:")
    print(comparison.to_string(index=False))
    
    # create a bar chart for accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies)
    plt.xlabel('model')
    plt.ylabel('accuracy (%)')
    plt.title('model accuracy comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    
    # create a scatter plot for accuracy vs training time
    plt.figure(figsize=(10, 6))
    plt.scatter(training_times, accuracies, s=100)
    for i, model in enumerate(model_names):
        plt.annotate(model, (training_times[i], accuracies[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('training time (s)')
    plt.ylabel('accuracy (%)')
    plt.title('accuracy vs. training time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_vs_time.png')
    plt.close()
    
    return comparison

#=====================
# feature extraction
#=====================

def extract_deep_features(model, image_tensor, layer_name=None):
    """
    extract deep features from a specific layer of the model - helps with analysis
    
    args:
        model: pytorch model
        image_tensor: input image tensor
        layer_name: name of the layer to extract features from (optional)
        
    returns:
        features: extracted features
    """
    # put model in eval mode
    model.eval()
    
    # dictionary to store features
    features = {}
    
    # hook function to extract features
    def hook_fn(module, input, output):
        features['output'] = output.detach().cpu()
    
    # register hook for the specified layer
    if layer_name:
        if hasattr(model, layer_name):
            getattr(model, layer_name).register_forward_hook(hook_fn)
        else:
            raise ValueError(f"layer {layer_name} not found in model")
    else:
        # use the penultimate layer for feature extraction by default
        if hasattr(model, 'features'):
            # for models with a features attribute (e.g., custom model)
            model.features[-1].register_forward_hook(hook_fn)
        elif hasattr(model, 'layer4'):
            # for resnet-like models
            model.layer4.register_forward_hook(hook_fn)
        elif hasattr(model, 'classifier'):
            # for models with classifier (e.g., efficientnet, mobilenet)
            model.classifier[0].register_forward_hook(hook_fn)
            
    # forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    return features['output']

#=====================
# feature visualization
#=====================

def visualize_feature_maps(feature_maps, num_features=8, save_path=None):
    """
    visualize feature maps from convolutional layers to understand what the model is learning
    
    args:
        feature_maps: feature maps tensor [batch, channels, height, width]
        num_features: number of feature maps to visualize
        save_path: path to save the visualization
    """
    # take the first sample if batch size > 1
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]
        
    # number of feature maps to plot
    n = min(num_features, feature_maps.shape[0])
    
    # create grid of feature maps
    fig, axes = plt.subplots(2, n//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n):
        feature_map = feature_maps[i].numpy()
        # normalize for visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'feature {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"feature map visualization saved to {save_path}")
        
    plt.show()

def plot_feature_importance(model, sample_image, class_idx=1, save_path=None):
    """
    visualize which parts of the image are important for the model's decision using gradientshap
    
    args:
        model: pytorch model
        sample_image: sample image tensor [1, C, H, W]
        class_idx: class index (1 for fake, 0 for real)
        save_path: path to save the visualization
    """
    # initialize gradientshap
    model.eval()
    gradient_shap = GradientShap(model)
    
    # create baseline (black image)
    baseline = torch.zeros_like(sample_image)
    
    # set the references for gradientshap
    input_references = torch.cat([baseline, torch.ones_like(sample_image)])
    
    # calculate attributions
    attributions = gradient_shap.attribute(sample_image, 
                                         n_samples=50,
                                         stdevs=0.2,
                                         baselines=input_references,
                                         target=class_idx)
    
    # convert to numpy and take absolute values
    attr_np = np.abs(attributions.detach().cpu().numpy()[0])
    
    # sum over channels to get a single heatmap
    heatmap = np.sum(attr_np, axis=0)
    
    # normalize for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # create the visualization
    plt.figure(figsize=(12, 5))
    
    # original image
    plt.subplot(1, 2, 1)
    img = sample_image[0].detach().cpu().numpy().transpose(1, 2, 0)
    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title("original image")
    plt.axis('off')
    
    # feature importance heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(label='feature importance')
    plt.title("feature importance")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"feature importance visualization saved to {save_path}")
        
    plt.show()
    
    return heatmap

#=====================
# explainable ai tools
#=====================

def generate_grad_cam(model, image_tensor, class_idx, layer_name=None):
    """
    generate grad-cam visualization to highlight important regions in the image
    
    args:
        model: pytorch model
        image_tensor: input image tensor [1, C, H, W]
        class_idx: target class index (1 for fake, 0 for real)
        layer_name: name of the layer to use for grad-cam
        
    returns:
        grad_cam_heatmap: grad-cam heatmap
    """
    # put model in eval mode
    model.eval()
    image_tensor.requires_grad_()
    
    # determine the target layer
    if layer_name:
        target_layer = getattr(model, layer_name)
    else:
        # use the last convolutional layer by default
        if hasattr(model, 'features') and isinstance(model.features[-3], torch.nn.Conv2d):
            target_layer = model.features[-3]
        elif hasattr(model, 'layer4'):
            # for resnet
            target_layer = model.layer4[-1].conv2
        else:
            raise ValueError("could not find a suitable convolutional layer")
    
    # activations and gradients storage
    activations = []
    gradients = []
    
    # hook functions
    def forward_hook_fn(module, input, output):
        activations.append(output)
        
    def backward_hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook_fn)
    backward_handle = target_layer.register_full_backward_hook(backward_hook_fn)
    
    # forward pass
    outputs = model(image_tensor)
    
    # clear gradients
    model.zero_grad()
    
    # backward pass with respect to target class
    if outputs.dim() > 1:
        outputs[0, class_idx].backward()
    else:
        outputs[class_idx].backward()
    
    # remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # get activations and gradients
    activations = activations[0].detach().cpu().numpy()[0]  # [C, H, W]
    gradients = gradients[0].detach().cpu().numpy()[0]      # [C, H, W]
    
    # global average pooling of gradients
    weights = np.mean(gradients, axis=(1, 2))  # [C]
    
    # create the class activation map
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]
    
    # apply relu to focus only on features that have a positive influence
    cam = np.maximum(cam, 0)
    
    # normalize for visualization
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

def visualize_grad_cam(model, image_tensor, original_image, class_labels, save_path=None):
    """
    create and visualize grad-cam for both classes (real and fake)
    
    args:
        model: pytorch model
        image_tensor: input image tensor [1, C, H, W]
        original_image: original pil image
        class_labels: list of class labels ['real', 'fake']
        save_path: path to save the visualization
    """
    # generate grad-cam for both classes
    cam_real = generate_grad_cam(model, image_tensor, 0)  # real class
    cam_fake = generate_grad_cam(model, image_tensor, 1)  # fake class
    
    # resize cam to original image size
    img_width, img_height = original_image.size
    cam_real = cv2.resize(cam_real, (img_width, img_height))
    cam_fake = cv2.resize(cam_fake, (img_width, img_height))
    
    # convert pil image to numpy array
    img_np = np.array(original_image)
    
    # create heatmap overlay for both classes
    heatmap_real = cv2.applyColorMap(np.uint8(255 * cam_real), cv2.COLORMAP_JET)
    heatmap_real = cv2.cvtColor(heatmap_real, cv2.COLOR_BGR2RGB)
    
    heatmap_fake = cv2.applyColorMap(np.uint8(255 * cam_fake), cv2.COLORMAP_JET)
    heatmap_fake = cv2.cvtColor(heatmap_fake, cv2.COLOR_BGR2RGB)
    
    # blend original image with heatmaps
    alpha = 0.7
    superimposed_real = heatmap_real * alpha + img_np * (1 - alpha)
    superimposed_real = np.uint8(superimposed_real)
    
    superimposed_fake = heatmap_fake * alpha + img_np * (1 - alpha)
    superimposed_fake = np.uint8(superimposed_fake)
    
    # create the visualization
    plt.figure(figsize=(15, 5))
    
    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("original image")
    plt.axis('off')
    
    # grad-cam for real class
    plt.subplot(1, 3, 2)
    plt.imshow(superimposed_real)
    plt.title(f"grad-cam: {class_labels[0]}")
    plt.axis('off')
    
    # grad-cam for fake class
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_fake)
    plt.title(f"grad-cam: {class_labels[1]}")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"grad-cam visualization saved to {save_path}")
        
    plt.show()

#=====================
# evaluation metrics
#=====================

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    create an enhanced confusion matrix visualization with seaborn
    
    args:
        conf_matrix: confusion matrix
        class_names: list of class names
        save_path: path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # use seaborn for better aesthetics
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('confusion matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"confusion matrix visualization saved to {save_path}")
        
    plt.show()

def plot_roc_curve(model, test_loader, device, save_path=None):
    """
    plot roc curve for the model to evaluate performance across thresholds
    
    args:
        model: pytorch model
        test_loader: dataloader for test data
        device: device to use for inference
        save_path: path to save the visualization
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of fake class
            all_labels.extend(labels.cpu().numpy())
    
    # calculate roc curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # plot roc curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'roc curve (auc = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('receiver operating characteristic (roc)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"roc curve saved to {save_path}")
        
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(model, test_loader, device, save_path=None):
    """
    plot precision-recall curve for the model - useful for imbalanced datasets
    
    args:
        model: pytorch model
        test_loader: dataloader for test data
        device: device to use for inference
        save_path: path to save the visualization
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of fake class
            all_labels.extend(labels.cpu().numpy())
    
    # calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    
    # plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'pr curve (ap = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"pr curve saved to {save_path}")
        
    plt.show()
    
    return avg_precision

#=====================
# feature analysis with t-SNE
#=====================

def extract_and_visualize_tsne(model, test_loader, device, perplexity=30, save_path=None):
    """
    extract features from the model and visualize them using t-sne dimensionality reduction
    
    args:
        model: pytorch model
        test_loader: dataloader for test data
        device: device to use for inference
        perplexity: perplexity parameter for t-sne
        save_path: path to save the visualization
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # extract features from the penultimate layer
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # extract features
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                # for models with separate feature extractor and classifier
                features = model.features(inputs)
                if isinstance(features, torch.Tensor):
                    features = torch.flatten(features, start_dim=1)
            elif hasattr(model, 'layer4') and hasattr(model, 'avgpool'):
                # for resnet-like models
                x = model.conv1(inputs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                features = model.layer4(x)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)
            else:
                # fallback: forward pass until the last layer
                for name, module in list(model.named_children())[:-1]:
                    inputs = module(inputs)
                features = torch.flatten(inputs, start_dim=1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # concatenate features and labels
    features_array = np.vstack(features_list)
    labels_array = np.concatenate(labels_list)
    
    # apply t-sne for dimensionality reduction
    print(f"applying t-sne with perplexity {perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_array)
    
    # visualize the 2d projection
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=labels_array, cmap='viridis', alpha=0.7, s=50)
    
    plt.colorbar(scatter, label='class')
    plt.title(f't-sne visualization of features (perplexity={perplexity})')
    plt.xlabel('t-sne component 1')
    plt.ylabel('t-sne component 2')
    
    # add a legend
    class_names = ['real', 'fake']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=scatter.cmap(scatter.norm(i)), 
                               markersize=10, label=name)
                     for i, name in enumerate(class_names)]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"t-sne visualization saved to {save_path}")
        
    plt.show()
    
    return features_tsne, labels_array

#=====================
# error analysis
#=====================

def analyze_failures(model, test_loader, device, class_names, num_samples=5, save_dir=None):
    """
    analyze and visualize failure cases to understand model limitations
    
    args:
        model: pytorch model
        test_loader: dataloader for test data
        device: device to use for inference
        class_names: list of class names
        num_samples: number of failure samples to visualize (per class)
        save_dir: directory to save the visualizations
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    model.eval()
    
    # store false positives and false negatives
    false_positives = []  # real images classified as fake
    false_negatives = []  # fake images classified as real
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # identify misclassified samples
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    # denormalize the image
                    img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    if label == 0 and pred == 1:  # real classified as fake
                        false_positives.append((img, outputs[i].cpu().numpy()))
                    elif label == 1 and pred == 0:  # fake classified as real
                        false_negatives.append((img, outputs[i].cpu().numpy()))
    
    # limit the number of samples to visualize
    false_positives = false_positives[:num_samples]
    false_negatives = false_negatives[:num_samples]
    
    # visualize false positives
    if false_positives:
        plt.figure(figsize=(15, 4 * len(false_positives)))
        for i, (img, logits) in enumerate(false_positives):
            plt.subplot(len(false_positives), 1, i+1)
            plt.imshow(img)
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
            plt.title(f"real image classified as fake\nprobabilities: real: {probs[0]:.3f}, fake: {probs[1]:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'false_positives.png'))
            print(f"false positives visualization saved to {os.path.join(save_dir, 'false_positives.png')}")
        
        plt.show()
    
    # visualize false negatives
    if false_negatives:
        plt.figure(figsize=(15, 4 * len(false_negatives)))
        for i, (img, logits) in enumerate(false_negatives):
            plt.subplot(len(false_negatives), 1, i+1)
            plt.imshow(img)
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
            plt.title(f"fake image classified as real\nprobabilities: real: {probs[0]:.3f}, fake: {probs[1]:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'false_negatives.png'))
            print(f"false negatives visualization saved to {os.path.join(save_dir, 'false_negatives.png')}")
        
        plt.show()
    
    print(f"found {len(false_positives)} false positives and {len(false_negatives)} false negatives")