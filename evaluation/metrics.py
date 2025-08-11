"""Evaluation utilities for DIET finetuning framework."""
import numpy as np
import torch
import time
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def zero_shot_eval(net, test_loader, num_classes, device, eval_id=None):
    """Evaluate model using zero-shot methods
    
    Args:
        net: The model to evaluate
        test_loader: DataLoader for test data
        num_classes: Number of classes in the dataset
        device: Device to run evaluation on
        eval_id: Optional identifier for the evaluation
        
    Returns:
        dict: Dictionary with zero-shot metrics
    """
    if eval_id is None:
        eval_id = int(time.time()) % 10000

    start_time = time.time()
    print("Extracting features for zero-shot evaluation...")

    features = []
    labels = []
    net.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting features"):
            # Unpack batch flexibly
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError("Unexpected batch structure")
            x = x.to(device)
            feat = net(x)
            features.append(feat.cpu().numpy())
            labels.append(y.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels).ravel()

    feature_hash = hash(str(features[:3].sum()))
    print(f"Feature hash: {feature_hash} (should change between evaluations)")
    print(f"Features extracted: {features.shape}, time: {time.time() - start_time:.2f}s")

    results = {}
    print("Running k-NN evaluation...")
    knn_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(features, labels)
    knn_pred = knn.predict(features)
    results["knn_acc"] = accuracy_score(labels, knn_pred)
    print(f"k-NN accuracy: {results['knn_acc']:.4f}, time: {time.time() - knn_time:.2f}s")

    print("Running k-means clustering evaluation...")
    kmeans_time = time.time()
    kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init=10)
    cluster_pred = kmeans.fit_predict(features)
    results["kmeans_ari"] = adjusted_rand_score(labels, cluster_pred)
    results["kmeans_nmi"] = normalized_mutual_info_score(labels, cluster_pred)
    print(f"k-means ARI: {results['kmeans_ari']:.4f}, NMI: {results['kmeans_nmi']:.4f}, time: {time.time() - kmeans_time:.2f}s")

    print("Running linear probe evaluation...")
    linear_time = time.time()
    X_train, X_test = features[:len(features)//2], features[len(features)//2:]
    y_train, y_test = labels[:len(labels)//2], labels[len(labels)//2:]
    linear_clf = torch.nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    linear_clf.train()
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    for epoch in range(50):
        optimizer.zero_grad()
        output = linear_clf(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    linear_clf.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        output = linear_clf(X_test_tensor)
        pred = output.argmax(dim=1).cpu().numpy()
        results["linear_acc"] = accuracy_score(y_test, pred)

    print(f"Linear probe accuracy: {results['linear_acc']:.4f}, time: {time.time() - linear_time:.2f}s")
    print(f"Total zero-shot evaluation time: {time.time() - start_time:.2f}s")
    return results

def evaluate_test_set(net, test_loader, device, W_probe):
    """Evaluate model on test set with probe classifier
    
    Args:
        net: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        W_probe: Probe classifier
        
    Returns:
        float: Test accuracy
    """
    print("\nStarting evaluation on test set:")
    net.eval()
    with torch.no_grad():
        run_acc_test = []
        for i, batch in enumerate(test_loader):
            # Flexible unpacking: works with (x, y) or (x, y, n)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, y, _ = batch
                elif len(batch) == 2:
                    x, y = batch
                else:
                    print(f"Warning: Unexpected test batch length ({len(batch)}) at batch {i}. Skipping.")
                    continue
            else:
                print(f"Warning: Unexpected test batch type at batch {i}. Skipping.")
                continue

            x = x.to(device)
            y = y.to(device)
            z = net(x)
            logits_probe = W_probe(z)
            
            # Adjust dimensions if necessary
            if y.dim() != logits_probe.argmax(1).dim():
                y = y.squeeze() if y.dim() > logits_probe.argmax(1).dim() else y.unsqueeze(0)
            batch_acc = torch.mean((y == logits_probe.argmax(1)).float()).item()
            run_acc_test.append(batch_acc)
            
            if i % 10 == 0:  # Print only every 10 batches to reduce verbosity
                print(f"Test Batch {i}: Accuracy={batch_acc:.4f}")
            
        test_acc = np.mean(run_acc_test) if run_acc_test else 0
        print(f"\nOverall Test Accuracy: {test_acc:.4f}")
    return test_acc

def create_zero_shot_progression_plot(metrics_history, tracked_epochs, metrics_list):
    """Create zero-shot metrics progression plot
    
    Args:
        metrics_history: Dictionary of metrics history
        tracked_epochs: List of epochs to track
        metrics_list: List of metric names to include
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=(15, 10))
    
    # Plot each metric's progression
    for i, metric in enumerate(metrics_list):
        ax = fig.add_subplot(2, 2, i+1)
        values = [metrics_history["zero_shot_metrics"][e][metric] for e in tracked_epochs]
        ax.plot(tracked_epochs, values, marker='o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric} Score')
        ax.set_title(f'Zero-shot {metric} Progression')
        ax.grid(True)
        
        # Add initial and final values as text annotations
        ax.annotate(f'{values[0]:.4f}', (tracked_epochs[0], values[0]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        ax.annotate(f'{values[-1]:.4f}', (tracked_epochs[-1], values[-1]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    fig.tight_layout()
    fig.suptitle('Zero-shot Metrics Progression During Training', fontsize=16)
    fig.subplots_adjust(top=0.9)
    
    return fig

def create_training_progress_plot(metrics_history):
    """Create training progress plot
    
    Args:
        metrics_history: Dictionary of metrics history
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=(15, 5))
    
    # Plot loss
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(metrics_history["train_loss_diet"], label="DIET Loss")
    ax1.plot(metrics_history["train_loss_probe"], label="Probe Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(metrics_history["train_acc"], label="Train Accuracy")
    ax2.plot(metrics_history["test_acc"], label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    # Plot zero-shot metrics
    ax3 = fig.add_subplot(1, 3, 3)
    
    # Get initial and final zero-shot metrics
    tracked_epochs = sorted(metrics_history["zero_shot_metrics"].keys())
    if len(tracked_epochs) >= 2:
        initial_epoch = tracked_epochs[0]
        final_epoch = tracked_epochs[-1]
        
        metrics = list(metrics_history["zero_shot_metrics"][initial_epoch].keys())
        x = range(len(metrics))
        width = 0.35
        
        # Plot bar chart comparing initial and final metrics
        ax3.bar(x, [metrics_history["zero_shot_metrics"][initial_epoch][m] for m in metrics], 
                width, label='Initial')
        ax3.bar([i + width for i in x], [metrics_history["zero_shot_metrics"][final_epoch][m] for m in metrics], 
                width, label='Final')
        ax3.set_xlabel("Metrics")
        ax3.set_ylabel("Score")
        ax3.set_title("Zero-Shot Performance")
        ax3.set_xticks([i + width/2 for i in x])
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True)
    
    fig.tight_layout()
    
    return fig

def calculate_metrics_improvements(initial_metrics, final_metrics):
    """Calculate improvements between initial and final metrics
    
    Args:
        initial_metrics: Dictionary with initial metric values
        final_metrics: Dictionary with final metric values
        
    Returns:
        dict: Dictionary with improvements and relative improvements
    """
    improvements = {}
    
    for metric in initial_metrics:
        initial = initial_metrics[metric]
        final = final_metrics[metric]
        change = final - initial
        relative_change = (change / initial) * 100 if initial > 0 else float('inf')
        
        improvements[f"improvement_{metric}"] = change
        improvements[f"relative_improvement_{metric}"] = relative_change
    
    # Calculate average improvement
    avg_improvement = np.mean([improvements[f"improvement_{m}"] for m in initial_metrics.keys()])
    avg_rel_improvement = np.mean([improvements[f"relative_improvement_{m}"] for m in initial_metrics.keys() 
                                 if improvements[f"relative_improvement_{m}"] != float('inf')])
    
    improvements["average_improvement"] = avg_improvement
    improvements["average_relative_improvement"] = avg_rel_improvement
    
    return improvements