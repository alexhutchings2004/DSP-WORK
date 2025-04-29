import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

#=====================
# Package Imports
#=====================
from deepfake_detection.data_loader import get_data_loaders, get_advanced_transformations, get_multimodal_data_loaders
from deepfake_detection.models import get_model, get_model_size
from deepfake_detection.train import train_model, evaluate_model
from deepfake_detection.utils import (
    set_seed, plot_metrics, save_metrics_to_csv, compare_models,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, 
    extract_and_visualize_tsne, analyze_failures
)

def main():
    #=====================
    # Parse Arguments
    #=====================
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--data_dir', type=str, default='.', help='Base directory path containing train, valid, and test folders')
    parser.add_argument('--models', type=str, nargs='+', default=['all'], 
                        choices=['all', 'resnet18', 'efficientnet'],
                        help='Model architectures to train (use "all" for all models)')
    parser.add_argument('--skip_models', type=str, nargs='+', default=[], 
                        help='Models to skip (useful for resuming training after some models are completed)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--face_detection', action='store_true', help='Enable face detection preprocessing')
    parser.add_argument('--feature_extraction', action='store_true', help='Enable feature extraction capabilities')
    parser.add_argument('--advanced_viz', action='store_true', help='Enable advanced visualizations')
    parser.add_argument('--noise_analysis', action='store_true', help='Enable noise pattern analysis')
    parser.add_argument('--freq_analysis', action='store_true', help='Enable frequency domain analysis')
    parser.add_argument('--tsne_viz', action='store_true', help='Create t-SNE visualizations of features')
    parser.add_argument('--multimodal', action='store_true', help='Use multimodal data loading')
    parser.add_argument('--modalities', type=str, nargs='+', default=['rgb', 'noise', 'freq'], 
                        help='Modalities for multimodal training')
    
    args = parser.parse_args()
    
    #=====================
    # Setup Environment
    #=====================
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Use GPU if available for faster training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #=====================
    # Data Loading
    #=====================
    # Get appropriate data loaders based on configuration
    if args.multimodal:
        print(f"Loading multimodal data with modalities: {args.modalities}")
        train_loader, val_loader, test_loader = get_multimodal_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            img_size=224,
            seed=args.seed,
            modalities=args.modalities
        )
    else:
        print("Loading standard data")
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            img_size=224,
            seed=args.seed,
            face_detection=args.face_detection,
            noise_analysis=args.noise_analysis,
            freq_analysis=args.freq_analysis
        )
    
    print(f"Dataset loaded. Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    #=====================
    # Model Selection
    #=====================
    # Determine which models to train based on args
    if 'all' in args.models:
        if args.multimodal:
            models_to_train = ['multimodal', 'dual_stream']
        else:
            models_to_train = ['resnet18', 'efficientnet']
    else:
        models_to_train = args.models
    
    # Filter out models to skip for resumed training scenarios
    models_to_train = [model for model in models_to_train if model not in args.skip_models]
    if models_to_train:
        print(f"Models to train: {', '.join(models_to_train)}")
    else:
        print("No models to train after applying skip filter.")
        return
    
    #=====================
    # Training Pipeline
    #=====================
    # Train all selected models sequentially and collect results
    results = {}
    start_time_all = time.time()
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f" Training {model_name.upper()} model")
        print(f"{'='*50}")
        
        # Train and evaluate the model
        result = train_and_evaluate_model(
            model_name, 
            train_loader, 
            val_loader, 
            test_loader, 
            device, 
            args
        )
        
        # Store results for comparison
        results[model_name] = result
        
        # Create advanced visualizations if enabled
        if args.advanced_viz:
            print(f"\nCreating advanced visualizations for {model_name}...")
            create_advanced_visualizations(
                model_name,
                result['model'],
                test_loader,
                device,
                os.path.join(args.output_dir, model_name),
                args
            )
    
    total_time = time.time() - start_time_all
    print(f"\nTotal training time for all models: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    #=====================
    # Results Analysis
    #=====================
    # Compare models and save comprehensive results
    comparison = compare_models(results)
    comparison.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
    
    # Generate comprehensive performance dashboard
    create_performance_dashboard(results, args.output_dir)
    
    print("\nTraining of all models completed successfully!")

def train_and_evaluate_model(model_name, train_loader, val_loader, test_loader, device, args):
    """Train and evaluate a single model with comprehensive metrics collection."""
    #=====================
    # Model Initialization
    #=====================
    # Initialize model with feature extraction if requested
    model = get_model(model_name, num_classes=2, pretrained=True, feature_extraction=args.feature_extraction)
    print(f"Model initialized: {model_name}")
    
    # Calculate and print model size for deployment considerations
    model_size = get_model_size(model)
    print(f"Model Size: {model_size:.2f} MB")
    
    # Move model to appropriate device (GPU/CPU)
    model = model.to(device)
    
    #=====================
    # Training Setup
    #=====================
    # Loss function and optimizer configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Create paths for model checkpoints and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}.pth")
    plot_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_metrics.png")
    csv_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_metrics.csv")
    
    #=====================
    # Training Process
    #=====================
    # Train the model and track training time
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=args.epochs, scheduler=scheduler, save_path=model_save_path
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Visualize and save training metrics
    plot_metrics(history, save_path=plot_path)
    save_metrics_to_csv(history, csv_path)
    
    #=====================
    # Model Evaluation
    #=====================
    # Evaluate the model on test data
    results = evaluate_model(model, test_loader, device)
    test_acc = results['accuracy']
    conf_matrix = np.array(results['confusion_matrix'])
    
    # Generate and save confusion matrix visualization
    cm_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    #=====================
    # Model Persistence
    #=====================
    # Save final model with comprehensive metadata
    final_model_path = os.path.join(args.output_dir, f"{model_name}_final.pth")
    torch.save({
        'model_name': model_name,
        'state_dict': model.state_dict(),
        'accuracy': test_acc,
        'class_names': ['Real', 'Fake'],
        'img_size': 224,
        'training_history': history,
        'confusion_matrix': conf_matrix.tolist(),
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Return comprehensive results dictionary
    return {
        'accuracy': test_acc,
        'training_time': training_time,
        'model_size': model_size,
        'history': history,
        'confusion_matrix': conf_matrix,
        'model': model,
        'model_path': final_model_path
    }

def create_advanced_visualizations(model_name, model, test_loader, device, output_dir, args):
    """Create advanced visualizations for model analysis and interpretability."""
    #=====================
    # Visualization Setup
    #=====================
    os.makedirs(output_dir, exist_ok=True)
    
    #=====================
    # Performance Curves
    #=====================
    # Create ROC curve for threshold selection analysis
    print("Creating ROC curve...")
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    roc_auc = plot_roc_curve(model, test_loader, device, save_path=roc_path)
    print(f"ROC curve saved (AUC = {roc_auc:.3f})")
    
    # Create Precision-Recall curve for imbalanced dataset analysis
    print("Creating Precision-Recall curve...")
    pr_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
    avg_precision = plot_precision_recall_curve(model, test_loader, device, save_path=pr_path)
    print(f"PR curve saved (AP = {avg_precision:.3f})")
    
    #=====================
    # Feature Analysis
    #=====================
    # Create t-SNE visualization for feature space analysis
    if args.tsne_viz:
        print("Creating t-SNE visualization (this may take a while)...")
        tsne_path = os.path.join(output_dir, f"{model_name}_tsne.png")
        features_tsne, labels = extract_and_visualize_tsne(model, test_loader, device, save_path=tsne_path)
        print("t-SNE visualization saved")
    
    #=====================
    # Error Analysis
    #=====================
    # Analyze and visualize failure cases for model improvement
    print("Analyzing failure cases...")
    failure_dir = os.path.join(output_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)
    analyze_failures(model, test_loader, device, ['Real', 'Fake'], save_dir=failure_dir)
    print("Failure analysis completed")

def create_performance_dashboard(results, output_dir):
    """Create a comprehensive performance dashboard comparing all models."""
    #=====================
    # Dashboard Preparation
    #=====================
    if not results:
        return
    
    dashboard_path = os.path.join(output_dir, "performance_dashboard.png")
    
    # Extract data from results dictionary
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    train_times = [results[m]['training_time'] / 60 for m in models]  # Convert to minutes
    model_sizes = [results[m]['model_size'] for m in models]
    
    #=====================
    # Dashboard Creation
    #=====================
    # Create comprehensive figure with multiple visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar chart of accuracies for direct performance comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(models, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # 2. Scatter plot of accuracy vs. training time for efficiency analysis
    plt.subplot(2, 2, 2)
    plt.scatter(train_times, accuracies, s=100)
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(model, (train_times[i], accuracies[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Training Time')
    plt.grid(True, alpha=0.3)
    
    # 3. Bubble chart of accuracy vs. model size for deployment considerations
    plt.subplot(2, 2, 3)
    # Use training time for bubble size
    sizes = [t * 10 for t in train_times]  # Scale up for visibility
    sc = plt.scatter(model_sizes, accuracies, s=sizes, alpha=0.6)
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(model, (model_sizes[i], accuracies[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Model Size (bubble size = training time)')
    plt.grid(True, alpha=0.3)
    
    # 4. Training/validation loss curves for convergence analysis
    plt.subplot(2, 2, 4)
    for model in models:
        history = results[model]['history']
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['val_loss'], label=f"{model}")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add an overall title
    plt.suptitle('Deepfake Detection Model Performance Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    #=====================
    # Dashboard Export
    #=====================
    # Save the comprehensive dashboard
    plt.savefig(dashboard_path)
    plt.close()
    
    print(f"Performance dashboard saved to {dashboard_path}")

if __name__ == '__main__':
    main()