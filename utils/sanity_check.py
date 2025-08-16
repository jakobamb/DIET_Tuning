import torch
import torchvision
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define the device for model execution
from config import DEVICE

# Import model getter functions (adjust paths if necessary)
# It's better practice to have these in their respective model files
# and import them here, e.g., from models.dinov2 import get_dinov2_model
# For now, assuming they might be accessible or defined globally/imported in main
# This part might need adjustment based on your final project structure.
# If get_dinov2_model etc. are defined *inside* main.py's functions, this won't work directly.
# Let's assume they are importable for now.
try:
    from models.dinov2 import get_dinov2_model
    from models.mae import get_mae_model
    from models.mambavision import get_mambavision_model
    from models.ijepa import get_ijepa_model
    from models.aim import get_aim_model
except ImportError as e:
    print(
        f"Warning: Could not import all model getter functions in sanity_check.py: {e}"
    )

    # Define dummy functions if needed, or handle the error appropriately
    def get_dinov2_model(device, model_size):
        raise NotImplementedError("DINOv2 getter not found")

    def get_mae_model(device, model_size):
        raise NotImplementedError("MAE getter not found")

    def get_mambavision_model(device, model_variant):
        raise NotImplementedError("MambaVision getter not found")

    def get_ijepa_model(device, model_size):
        raise NotImplementedError("IJEPA getter not found")

    def get_aim_model(device, model_size):
        raise NotImplementedError("AIM getter not found")


def unified_sanity_check(
    model_type,
    model_size=None,
    model_variant=None,
    expected_threshold=None,  # Now optional, will be set based on model_type
    batch_size=None,  # Now optional, will be set based on model_type
    k_values=None,
    num_workers=0,
    log_to_wandb=True,  # Added parameter to control W&B logging
):
    """
    Unified sanity check with integrated W&B logging: Evaluate model's zero-shot
    performance on CIFAR10 using k-NN.

    Args:
        model_type (str): Type of model ("dinov2", "mae", "mambavision", "ijepa", "aim")
        model_size (str, optional): Model size for relevant models. Defaults based on model_type.
        model_variant (str, optional): Model variant (for mambavision). Defaults based on model_type.
        expected_threshold (float, optional): Expected accuracy threshold. Defaults based on model_type.
        batch_size (int, optional): Batch size for data loading. Defaults based on model_type.
        k_values (list, optional): List of k values to test. Defaults to [1, 5, 20, 50, 100, 200].
        num_workers (int, optional): Number of workers for data loader. Defaults to 0.
        log_to_wandb (bool, optional): Whether to log results to W&B. Defaults to True.

    Returns:
        dict: Results containing accuracies, k values, best accuracy, and best k, or None if failed.
    """
    if k_values is None:
        k_values = [1, 5, 20, 50, 100, 200]

    # Set model-specific parameters based on model type
    model_defaults = {
        "dinov2": {
            "model_size": "small",
            "expected_threshold": 0.91,
            "batch_size": 256,
        },
        "mae": {"model_size": "base", "expected_threshold": 0.85, "batch_size": 256},
        "mambavision": {
            "model_variant": "T",
            "expected_threshold": 0.85,
            "batch_size": 64,
        },
        "ijepa": {
            "model_size": "b16_1k",  # Example, adjust if needed
            "expected_threshold": 0.85,
            "batch_size": 64,
        },
        "aim": {
            "model_size": "600M",  # Example, adjust if needed
            "expected_threshold": 0.75,
            "batch_size": 256,
        },
    }

    # Apply default parameters if not provided
    if model_type in model_defaults:
        defaults = model_defaults[model_type]
        if model_size is None and "model_size" in defaults:
            model_size = defaults["model_size"]
        if model_variant is None and "model_variant" in defaults:
            model_variant = defaults["model_variant"]
        if expected_threshold is None and "expected_threshold" in defaults:
            expected_threshold = defaults["expected_threshold"]
        if batch_size is None and "batch_size" in defaults:
            batch_size = defaults["batch_size"]

    # Set some safe defaults if model type is not recognized
    if expected_threshold is None:
        expected_threshold = 0.85
    if batch_size is None:
        batch_size = 128

    print("\\n" + "=" * 70)
    print(f"SANITY CHECK: {model_type.upper()} ZERO-SHOT k-NN ON CIFAR10")
    print(f"Model Size/Variant: {model_size or model_variant}")
    print("=" * 70)

    # Initialize W&B tracking
    run = None
    if log_to_wandb:
        if wandb.run is None:
            # No active run, create one for this sanity check
            run_name = f"sanity_check_{model_type}"
            if model_size:
                run_name += f"_{model_size}"
            elif model_variant:
                run_name += f"_{model_variant}"

            try:
                run = wandb.init(
                    project="DIET-Finetuning-SanityChecks",
                    name=run_name,
                    config={
                        "model_type": model_type,
                        "model_size": model_size,
                        "model_variant": model_variant,
                        "expected_threshold": expected_threshold,
                        "batch_size": batch_size,
                        "k_values": k_values,
                    },
                    reinit=True,
                )  # Allow reinitialization if needed
                print(f"WandB initialized for sanity check: {run.name}")
            except Exception as e:
                print(f"WandB initialization failed for sanity check: {e}")
                log_to_wandb = False  # Disable logging if init fails
        else:
            # Use existing run
            run = wandb.run
            print(f"Using existing WandB run: {run.name}")
            # Log config to existing run
            try:
                run.config.update(
                    {
                        f"sanity_{model_type}_model_size": model_size,
                        f"sanity_{model_type}_model_variant": model_variant,
                        f"sanity_{model_type}_expected_threshold": expected_threshold,
                        f"sanity_{model_type}_batch_size": batch_size,
                        f"sanity_{model_type}_k_values": k_values,
                    },
                    allow_val_change=True,
                )
            except Exception as e:
                print(f"Failed to update WandB config for sanity check: {e}")

    # Load CIFAR10 dataset
    try:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

        cifar_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        cifar_test = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # Create data loaders
        train_loader = DataLoader(
            cifar_train, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    except Exception as e:
        error_msg = f"Failed to load CIFAR10 dataset: {e}"
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    # Create and load the appropriate model
    print(f"Loading fresh {model_type} model...")
    sanity_model = None
    embedding_dim = None
    try:
        if model_type == "dinov2":
            sanity_model, embedding_dim = get_dinov2_model(
                DEVICE, model_size=model_size
            )
        elif model_type == "mae":
            sanity_model, embedding_dim = get_mae_model(DEVICE, model_size=model_size)
        elif model_type == "mambavision":
            # MambaVision uses model_variant, which we mapped to model_size for consistency here
            sanity_model, embedding_dim = get_mambavision_model(
                DEVICE, model_variant=model_size
            )
        elif model_type == "ijepa":
            sanity_model, embedding_dim = get_ijepa_model(DEVICE, model_size=model_size)
        elif model_type == "aim":
            sanity_model, embedding_dim = get_aim_model(DEVICE, model_size=model_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if sanity_model is None or embedding_dim is None:
            raise ValueError("Model loading returned None.")

        sanity_model.eval()  # Set to evaluation mode

    except Exception as e:
        error_msg = f"Failed to load model '{model_type}' with size/variant '{model_size or model_variant}': {e}"
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    # Special case for I-JEPA: check if we need to run both kNN and linear probe
    run_linear_probe = model_type == "ijepa"
    linear_accuracy = None

    # Extract features from training set
    print("Extracting features from CIFAR10 training set...")
    train_features = []
    train_labels = []

    try:
        with torch.no_grad():
            for x, y in tqdm(train_loader, desc="Extracting train features"):
                x = x.to(DEVICE)
                feat = sanity_model(x)
                train_features.append(feat.cpu().numpy())
                train_labels.append(y.numpy())
    except Exception as e:
        error_msg = f"Error during training feature extraction: {e}"
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    if not train_features:
        error_msg = "No training features were extracted. Sanity check failed."
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)

    # Extract features from test set
    print("Extracting features from CIFAR10 test set...")
    test_features = []
    test_labels = []

    try:
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Extracting test features"):
                x = x.to(DEVICE)
                feat = sanity_model(x)
                test_features.append(feat.cpu().numpy())
                test_labels.append(y.numpy())
    except Exception as e:
        error_msg = f"Error during test feature extraction: {e}"
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    if not test_features:
        error_msg = "No test features were extracted. Sanity check failed."
        print(error_msg)
        if log_to_wandb and run:
            run.log({f"sanity_check_{model_type}_error": error_msg})
        return None

    test_features = np.vstack(test_features)
    test_labels = np.concatenate(test_labels)

    print(
        f"Features extracted: {train_features.shape} train, {test_features.shape} test"
    )

    # Normalize features (important for k-NN)
    train_features_normalized = train_features / np.linalg.norm(
        train_features, axis=1, keepdims=True
    )
    test_features_normalized = test_features / np.linalg.norm(
        test_features, axis=1, keepdims=True
    )

    # Run k-NN evaluation
    if run_linear_probe:
        print("\\n" + "=" * 50)
        print("k-NN EVALUATION")
        print("=" * 50)

    print("\\nEvaluating k-NN performance:")
    print("-" * 50)
    print(f"{'k value':<10} {'Accuracy':<10}")
    print("-" * 50)

    best_acc = 0
    best_k = 0
    accuracies = []

    # Create a W&B table for k-NN results
    knn_table = None
    if log_to_wandb and run:
        try:
            knn_table = wandb.Table(columns=["k_value", "accuracy"])
        except Exception as e:
            print(f"Failed to create WandB table for k-NN: {e}")

    for k in k_values:
        try:
            knn = KNeighborsClassifier(
                n_neighbors=k, metric="cosine"
            )  # Use cosine for normalized features
            knn.fit(train_features_normalized, train_labels)
            predictions = knn.predict(test_features_normalized)
            accuracy = accuracy_score(test_labels, predictions)
            accuracies.append(accuracy)
            print(f"{k:<10} {accuracy*100:.2f}%")

            # Log to W&B table
            if knn_table is not None:
                knn_table.add_data(k, accuracy * 100)  # Convert to percentage

            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k
        except Exception as e:
            print(f"Error during k-NN evaluation for k={k}: {e}")
            accuracies.append(0.0)  # Append 0 if error occurs

    print("-" * 50)

    # Log the k-NN table to W&B
    if knn_table is not None and log_to_wandb and run:
        try:
            run.log({f"sanity_check_{model_type}_knn_results": knn_table})
        except Exception as e:
            print(f"Failed to log k-NN table to WandB: {e}")

    # Linear probe evaluation for I-JEPA
    if run_linear_probe:
        print("\\n" + "=" * 50)
        print("LINEAR PROBE EVALUATION")
        print("=" * 50)

        try:
            # Convert features to PyTorch tensors
            train_features_tensor = torch.FloatTensor(train_features).to(DEVICE)
            train_labels_tensor = torch.LongTensor(train_labels).to(DEVICE)
            test_features_tensor = torch.FloatTensor(test_features).to(DEVICE)
            test_labels_tensor = torch.LongTensor(test_labels).to(DEVICE)

            # Set up linear probe
            num_classes = 10  # CIFAR10 has 10 classes
            linear_probe = nn.Linear(embedding_dim, num_classes).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)

            # Train linear probe
            num_epochs = 50
            linear_batch_size = 1024
            linear_probe.train()

            # Prepare data for batch training
            dataset = TensorDataset(train_features_tensor, train_labels_tensor)
            loader = DataLoader(dataset, batch_size=linear_batch_size, shuffle=True)

            # Create list to track loss for W&B
            loss_history = [] if log_to_wandb and run else None

            print("Training linear probe...")
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_features, batch_labels in loader:
                    optimizer.zero_grad()
                    logits = linear_probe(batch_features)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                epoch_loss = total_loss / len(loader)
                if loss_history is not None:
                    loss_history.append(epoch_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # Log linear probe training curve
            if loss_history is not None and log_to_wandb and run:
                try:
                    loss_table = wandb.Table(
                        data=[[i, loss] for i, loss in enumerate(loss_history)],
                        columns=["epoch", "loss"],
                    )
                    run.log(
                        {
                            f"sanity_check_{model_type}_linear_probe_loss": wandb.plot.line(
                                table=loss_table,
                                x="epoch",
                                y="loss",
                                title="Linear Probe Training Loss",
                            )
                        }
                    )
                except Exception as e:
                    print(f"Failed to log linear probe loss curve to WandB: {e}")

            # Evaluate linear probe
            linear_probe.eval()
            with torch.no_grad():
                logits = linear_probe(test_features_tensor)
                predictions = logits.argmax(dim=1).cpu().numpy()
                linear_accuracy = accuracy_score(test_labels, predictions)

            print(f"Linear probe accuracy: {linear_accuracy*100:.2f}%")

            # Log linear probe accuracy
            if log_to_wandb and run:
                try:
                    run.log(
                        {
                            f"sanity_check_{model_type}_linear_probe_acc": linear_accuracy
                            * 100
                        }
                    )
                except Exception as e:
                    print(f"Failed to log linear probe accuracy to WandB: {e}")

        except Exception as e:
            print(f"Error during linear probe evaluation: {e}")
            linear_accuracy = None  # Ensure it's None if error occurs

    # Determine if sanity check passed
    passed_check = best_acc >= expected_threshold
    status = "PASSED ✓" if passed_check else "FAILED ✗"

    print(f"\\nBest k-NN accuracy: {best_acc*100:.2f}% (k={best_k})")
    print(f"Sanity check status: {status}")
    print(
        f"Expected accuracy: >{expected_threshold*100}%, Achieved: {best_acc*100:.2f}%"
    )
    print("=" * 70)

    # Log summary results to W&B
    summary_table = None
    if log_to_wandb and run:
        try:
            run.log(
                {
                    f"sanity_check_{model_type}_best_acc": best_acc * 100,
                    f"sanity_check_{model_type}_best_k": best_k,
                    f"sanity_check_{model_type}_passed": passed_check,
                }
            )

            # Create summary table
            summary_table = wandb.Table(
                columns=[
                    "model",
                    "method",
                    "best_accuracy",
                    "best_k",
                    "threshold",
                    "passed_check",
                ]
            )

            # Add k-NN row
            summary_table.add_data(
                model_type,
                "k-NN",
                best_acc * 100,
                best_k,
                expected_threshold * 100,
                "✓" if passed_check else "✗",
            )

            # Add linear probe row if applicable
            if run_linear_probe and linear_accuracy is not None:
                linear_passed = linear_accuracy >= expected_threshold
                summary_table.add_data(
                    model_type,
                    "Linear Probe",
                    linear_accuracy * 100,
                    "N/A",
                    expected_threshold * 100,
                    "✓" if linear_passed else "✗",
                )

            # Log the summary table
            run.log({f"sanity_check_{model_type}_summary": summary_table})
        except Exception as e:
            print(f"Failed to log summary results/table to WandB: {e}")

    # Create visualization plot
    try:
        plt.figure(figsize=(10, 6))
        if run_linear_probe and linear_accuracy is not None:
            plt.figure(figsize=(15, 6))

            # Plot k-NN results
            plt.subplot(1, 2, 1)
            plt.plot(
                k_values, [acc * 100 for acc in accuracies], marker="o", linewidth=2
            )
            plt.axhline(
                y=expected_threshold * 100,
                color="r",
                linestyle="--",
                label=f"{expected_threshold*100}% threshold",
            )
            plt.xlabel("k value")
            plt.ylabel("Accuracy (%)")
            plt.title(f"{model_type.upper()} Zero-Shot k-NN Performance on CIFAR10")
            plt.grid(True)
            plt.legend()
            plt.xticks(k_values)

            # Plot comparison of methods
            plt.subplot(1, 2, 2)
            methods = ["k-NN (best)", "Linear Probe"]
            method_accuracies = [best_acc * 100, linear_accuracy * 100]

            plt.bar(methods, method_accuracies, color=["blue", "orange"])
            plt.ylabel("Accuracy (%)")
            plt.title("Zero-Shot Evaluation Methods Comparison")
            plt.grid(axis="y", alpha=0.3)
            plt.axhline(
                y=expected_threshold * 100,
                color="r",
                linestyle="--",
                label=f"{expected_threshold*100}% threshold",
            )
            plt.legend()

            # Add text on top of bars
            for i, v in enumerate(method_accuracies):
                plt.text(i, v + 1, f"{v:.2f}%", ha="center")
        else:
            # Only k-NN plot
            plt.plot(
                k_values, [acc * 100 for acc in accuracies], marker="o", linewidth=2
            )
            plt.axhline(
                y=expected_threshold * 100,
                color="r",
                linestyle="--",
                label=f"Expected threshold ({expected_threshold*100}%)",
            )
            plt.xlabel("k value")
            plt.ylabel("Accuracy (%)")
            plt.title(f"{model_type.upper()} Zero-Shot k-NN Performance on CIFAR10")
            plt.grid(True)
            plt.legend()
            plt.xticks(k_values)

        plt.tight_layout()

        # Log the figure to W&B
        if log_to_wandb and run:
            try:
                run.log({f"sanity_check_{model_type}_plot": wandb.Image(plt)})
            except Exception as e:
                print(f"Failed to log plot to WandB: {e}")

        # Close the plot to prevent display issues in non-interactive environments
        plt.close()

    except Exception as e:
        print(f"Error creating/logging plot: {e}")

    # Prepare return value
    results = {
        "accuracies": accuracies,
        "k_values": k_values,
        "best_acc": best_acc,
        "best_k": best_k,
        "passed_check": passed_check,
        "expected_threshold": expected_threshold,
    }

    if run_linear_probe and linear_accuracy is not None:
        results["linear_probe_acc"] = linear_accuracy

    # Finish the temporary W&B run if one was created
    if (
        log_to_wandb
        and run
        and wandb.run is run
        and run.name.startswith("sanity_check_")
    ):
        try:
            run.finish()
            print("Finished temporary WandB run for sanity check.")
        except Exception as e:
            print(f"Error finishing temporary WandB run: {e}")

    return results
