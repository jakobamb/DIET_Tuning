"""Dataset utilities for DIET finetuning framework."""
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.notebook import tqdm
from torchvision.transforms import v2

# Try to import MedMNIST
try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    print("MedMNIST not found, install with: pip install medmnist")
    MEDMNIST_AVAILABLE = False

class DatasetWithIndices(Dataset):
    """Wrapper that assigns random diet classes to dataset samples."""
    def __init__(self, dataset, num_diet_classes=200):
        self.dataset = dataset
        self.num_diet_classes = num_diet_classes
        # Assign each sample to one of num_diet_classes
        self.class_assignments = torch.randint(0, num_diet_classes, (len(dataset),))
        
    def __getitem__(self, n):
        # Convert tensor index to int if needed
        if isinstance(n, torch.Tensor):
            n = int(n.item())
        
        # Get sample from wrapped dataset
        item = self.dataset[n]
        
        # Handle different return formats
        if isinstance(item, tuple) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x = item
            y = torch.tensor(0)  # Default label if not provided
        
        # Ensure label is a proper tensor with correct dimension
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        # Make sure y has the right dimension (not a scalar)
        if y.dim() == 0:
            y = y.view(1)
        
        # Ensure diet class is a proper tensor with correct dimension
        diet_class = self.class_assignments[n]
        if diet_class.dim() == 0:
            diet_class = diet_class.view(1)
        
        return x, y, diet_class
        
    def __len__(self):
        return int(len(self.dataset))

class HFImageDataset(Dataset):
    """Wrapper for HuggingFace datasets."""
    def __init__(self, hf_dataset, transform=None, input_size=224):
        self.dataset = hf_dataset
        self.transform = transform
        self.input_size = input_size
        self.resize = transforms.Resize((input_size, input_size))
        
    def __len__(self):
        return int(len(self.dataset))
        
    def __getitem__(self, idx):
        # Convert idx to an integer if needed.
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        item = self.dataset[idx]
        image = item['image']
        label = item.get('label', item.get('labels'))
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Warning: Unexpected image format at index {idx}: {e}")
        image = self.resize(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class RobustGalaxyDataset(torch.utils.data.Dataset):
    """A robust dataset class specifically for Galaxy datasets that guarantees consistent tensor dimensions."""
    def __init__(self, hf_dataset, transform=None, diet_classes=100, limit_samples=None):
        """A robust dataset class that guarantees consistent tensor dimensions"""
        self.dataset = hf_dataset
        self.transform = transform
        
        # Limit samples if requested
        if limit_samples is not None and limit_samples < len(hf_dataset):
            indices = torch.randperm(len(hf_dataset))[:limit_samples].tolist()
            self.indices = indices
        else:
            self.indices = list(range(len(hf_dataset)))
        
        # Create diet class assignments - one per sample
        self.diet_classes = torch.randint(0, diet_classes, (len(self.indices),))
        
        # Create resize transform to ensure consistent image sizes
        self.resize = torchvision.transforms.Resize((256, 256))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the sample using our saved indices
        original_idx = self.indices[idx]
        sample = self.dataset[original_idx]
        
        # Get image and label
        image = sample['image']
        label = torch.tensor([sample['label']], dtype=torch.long)  # Create as 1D tensor
        
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except:
                print(f"Warning: Could not convert image to PIL at index {idx}")
        
        # Resize to ensure consistent dimensions
        image = self.resize(image)
        
        # Apply additional transforms
        if self.transform:
            image = self.transform(image)
        
        # Get diet class for this sample - ensure it's a 1D tensor
        diet_class = torch.tensor([self.diet_classes[idx].item()], dtype=torch.long)
        
        return image, label, diet_class

def get_dataset(dataset_name="cifar10", root='./data'):
    """
    Load the specified dataset with predetermined statistics.
    """
    # Predetermined mean and std values for common datasets
    dataset_stats = {
        "cifar10": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
            "input_size": 32,
            "is_rgb": True
        },
        "cifar100": {
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
            "input_size": 32,
            "is_rgb": True
        },
        "food101": {
            "mean": (0.485, 0.456, 0.406),  # ImageNet stats - good baseline for natural food images
            "std": (0.229, 0.224, 0.225),   # Food images have similar distribution to ImageNet
            "input_size": 224,  # Standard size for food classification (rescaled from 512)
            "is_rgb": True
        },
        "fgvc_aircraft": {
            "mean": (0.485, 0.456, 0.406),  # ImageNet stats - good baseline for aircraft images
            "std": (0.229, 0.224, 0.225),   # Aircraft images are natural outdoor scenes similar to ImageNet
            "input_size": 224,  # Standard size for fine-grained classification
            "is_rgb": True
        },
        "pathmnist": {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "input_size": 28,
            "is_rgb": True
        },
        "chestmnist": {
            "mean": (0.4984),
            "std": (0.2483),
            "input_size": 28,
            "is_rgb": False
        },
        "dermamnist": {
            "mean": (0.7634, 0.5423, 0.5698),
            "std": (0.0841, 0.1246, 0.1043),
            "input_size": 28,
            "is_rgb": True
        },
        "octmnist": {
            "mean": (0.1778),
            "std": (0.1316),
            "input_size": 28,
            "is_rgb": False
        },
        "pneumoniamnist": {
            "mean": (0.5060),
            "std": (0.2537),
            "input_size": 28,
            "is_rgb": False
        },
        "plantnet300k": {
            "mean": (0.485, 0.456, 0.406),  # ImageNet stats as starting point
            "std": (0.229, 0.224, 0.225),
            "input_size": 224,  # PlantNet images are resized to 224
            "is_rgb": True
        },
        "galaxy10_decals": {
            "mean": (0.097, 0.097, 0.097),  # Approximate for astronomy images (dark background)
            "std": (0.174, 0.164, 0.156),   # Astronomical images have different distribution
            "input_size": 256,  # Original image size is 256x256
            "is_rgb": True
        },
        "crop14_balance": {
            # Based on the dataset card, images are rescaled to a maximum side length of 512.
            "mean": (0.485, 0.456, 0.406),  # Using ImageNet stats as a placeholder
            "std": (0.229, 0.224, 0.225),
            "input_size": 512,
            "is_rgb": True
        }
    }
    
    # Load dataset based on name
    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True)
        num_classes = 10
        
    elif dataset_name.lower() == "cifar100":
        # Option 1: Use PyTorch's built-in CIFAR-100
        try:
            train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
            test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
            num_classes = 100
            print("CIFAR-100 loaded from PyTorch datasets")
            print("Dataset: 60,000 32x32 color images in 100 classes")
            print("Training samples: 50,000, Test samples: 10,000")
            print("Classes grouped into 20 superclasses with 5 classes each")
        except Exception as e:
            # Option 2: Fallback to HuggingFace dataset
            print(f"PyTorch CIFAR-100 failed ({e}), trying HuggingFace dataset...")
            try:
                from datasets import load_dataset
                
                print("Loading CIFAR-100 dataset from HuggingFace (randall-lab/cifar100)...")
                
                # Load the dataset
                dataset = load_dataset("randall-lab/cifar100", trust_remote_code=True)
                
                num_classes = 100
                input_size = dataset_stats["cifar100"]["input_size"]
                
                # Create train and test datasets
                train_dataset = HFImageDataset(dataset['train'], input_size=input_size)
                test_dataset = HFImageDataset(dataset['test'], input_size=input_size)
                
                print(f"CIFAR-100 loaded from HuggingFace: {len(train_dataset)} training, {len(test_dataset)} test samples")
                print(f"Number of classes: {num_classes}")
                print("Dataset contains 100 classes grouped into 20 superclasses:")
                print("- Examples: apple, aquarium fish, baby, bear, beaver, bed, bee, etc.")
                
            except ImportError:
                raise ImportError("HuggingFace datasets library is required for CIFAR-100 fallback. Install with 'pip install datasets'")
            except Exception as e:
                raise ValueError(f"Error loading CIFAR-100 dataset from both PyTorch and HuggingFace: {e}")
    
    elif dataset_name.lower() == "food101":
        # Food-101 dataset from HuggingFace with robust error handling
        try:
            from datasets import load_dataset
            import os
            
            print("Loading Food-101 dataset from HuggingFace (randall-lab/food101)...")
            print("Note: Food-101 is a large dataset (~5GB). This may take a while to download.")
            
            # Try to load with increased timeout and retry settings
            try:
                # Load the dataset with custom download configuration
                dataset = load_dataset(
                    "randall-lab/food101", 
                    trust_remote_code=True,
                    # Add download configuration for better reliability
                    download_config={
                        'resume_download': True,  # Resume interrupted downloads
                        'max_retries': 3,         # Retry failed downloads
                    }
                )
                
                num_classes = 101
                input_size = dataset_stats["food101"]["input_size"]
                
                # Create train and test datasets
                train_dataset = HFImageDataset(dataset['train'], input_size=input_size)
                test_dataset = HFImageDataset(dataset['test'], input_size=input_size)
                
                print(f"Food-101 loaded from HuggingFace: {len(train_dataset)} training, {len(test_dataset)} test samples")
                print(f"Number of classes: {num_classes}")
                print("Dataset: 101,000 images of food in 101 categories")
                print("Training samples: 75,750, Test samples: 25,250")
                print("Examples: apple_pie, baby_back_ribs, baklava, beef_carpaccio, etc.")
                print(f"Images resized to {input_size}x{input_size} (from max 512px)")
                
            except Exception as download_error:
                print(f"HuggingFace download failed: {download_error}")
                print("\nAlternative solutions:")
                print("1. Try again later (network issues)")
                print("2. Use a different dataset (cifar100, fgvc_aircraft)")
                print("3. Manually download Food-101 from: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/")
                print("\nFor now, falling back to CIFAR-100...")
                
                # Fallback to CIFAR-100 for immediate testing
                try:
                    train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
                    test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
                    num_classes = 100
                    print("Successfully loaded CIFAR-100 as fallback")
                    print("Dataset: 60,000 32x32 color images in 100 classes")
                    print("Training samples: 50,000, Test samples: 10,000")
                except Exception as fallback_error:
                    raise ValueError(f"Both Food-101 and CIFAR-100 fallback failed: {fallback_error}")
            
        except ImportError:
            raise ImportError("HuggingFace datasets library is required for Food-101. Install with 'pip install datasets'")
        except Exception as e:
            print(f"Food-101 loading failed: {e}")
            raise ValueError(f"Error loading Food-101 dataset: {e}")
        
    elif dataset_name.lower() == "fgvc_aircraft":
        # FGVC-Aircraft dataset from HuggingFace
        try:
            from datasets import load_dataset
            
            print("Loading FGVC-Aircraft dataset from HuggingFace (randall-lab/fgvc-aircraft)...")
            
            # Load the dataset
            dataset = load_dataset("randall-lab/fgvc-aircraft", trust_remote_code=True)
            
            num_classes = 100
            input_size = dataset_stats["fgvc_aircraft"]["input_size"]
            
            # Create train and test datasets - use validation as test since it has 3-way split
            train_dataset = HFImageDataset(dataset['train'], input_size=input_size)
            # Use validation split as test set (both val and test have 3,333 samples each)
            test_dataset = HFImageDataset(dataset['validation'], input_size=input_size)
            
            print(f"FGVC-Aircraft loaded from HuggingFace: {len(train_dataset)} training, {len(test_dataset)} test samples")
            print(f"Number of classes: {num_classes}")
            print("Dataset: 10,000 images of aircraft in 100 fine-grained model variants")
            print("Training samples: 3,334, Validation/Test samples: 3,333 each")
            print("Fine-grained classification: different aircraft models (not just types)")
            print(f"Images resized to {input_size}x{input_size} (from variable resolution)")
            
        except ImportError:
            raise ImportError("HuggingFace datasets library is required for FGVC-Aircraft. Install with 'pip install datasets'")
        except Exception as e:
            raise ValueError(f"Error loading FGVC-Aircraft dataset: {e}")
        
    elif MEDMNIST_AVAILABLE and dataset_name.lower() in INFO.keys():
        data_flag = dataset_name.lower()
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        # Get dataset information
        num_classes = len(info['label'])
        
        train_dataset = DataClass(split='train', download=True, root=root)
        test_dataset = DataClass(split='test', download=True, root=root)
        
        print(f"Dataset: {info['description']}")
        print(f"Task: {info['task']}")
        print(f"Number of classes: {num_classes}")
    
    elif dataset_name.lower() == "plantnet300k":
        # For PlantNet300K, we'll use HuggingFace datasets
        try:
            from datasets import load_dataset
            
            print("Loading PlantNet300K dataset from HuggingFace...")
            
            # Load the dataset
            dataset = load_dataset("mikehemberger/plantnet300K")
            
            # Number of classes is 85 according to the dataset card
            num_classes = 85
            
            # Get the input_size from stats
            input_size = dataset_stats["plantnet300k"]["input_size"]
            
            # Create train and test datasets with consistent sizing
            train_dataset = HFImageDataset(dataset['train'], input_size=input_size)
            
            # Use validation set as test set
            if 'validation' in dataset:
                test_dataset = HFImageDataset(dataset['validation'], input_size=input_size)
            else:
                test_dataset = HFImageDataset(dataset['test'], input_size=input_size)
            
            print(f"PlantNet300K loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")
            print(f"Number of classes: {num_classes}")
            print(f"All images will be resized to {input_size}x{input_size}")
            
        except ImportError:
            raise ImportError("HuggingFace datasets library is required for PlantNet300K. Install with 'pip install datasets'")
        except Exception as e:
            raise ValueError(f"Error loading PlantNet300K dataset: {e}")
    
    elif dataset_name.lower() == "galaxy10_decals":
        # For Galaxy10 DECals, we'll use HuggingFace datasets
        try:
            from datasets import load_dataset
            
            print("Loading Galaxy10 DECals dataset from HuggingFace...")
            
            # Load the dataset
            dataset = load_dataset("matthieulel/galaxy10_decals")
            
            # Number of classes is 10 according to the dataset card
            num_classes = 10
            
            # Get the input_size from stats
            input_size = dataset_stats["galaxy10_decals"]["input_size"]
            
            # Create train and test datasets with consistent sizing
            train_dataset = HFImageDataset(dataset['train'], input_size=input_size)
            test_dataset = HFImageDataset(dataset['test'], input_size=input_size)
            
            print(f"Galaxy10 DECals loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")
            print(f"Number of classes: {num_classes}")
            print(f"All images will be resized to {input_size}x{input_size}")
            print("Galaxy class labels:")
            print("0: Disturbed Galaxies")
            print("1: Merging Galaxies")
            print("2: Round Smooth Galaxies")
            print("3: In-between Round Smooth Galaxies") 
            print("4: Cigar Shaped Smooth Galaxies")
            print("5: Barred Spiral Galaxies")
            print("6: Unbarred Tight Spiral Galaxies")
            print("7: Unbarred Loose Spiral Galaxies")
            print("8: Edge-on Galaxies without Bulge")
            print("9: Edge-on Galaxies with Bulge")
            
        except ImportError:
            raise ImportError("HuggingFace datasets library is required for Galaxy10 DECals. Install with 'pip install datasets'")
        except Exception as e:
            raise ValueError(f"Error loading Galaxy10 DECals dataset: {e}")


    elif dataset_name.lower() == "crop14_balance":
        try:
            from datasets import load_dataset
            print("Loading crop14_balance dataset from Hugging Face (gary109/crop14_balance)...")
            dataset = load_dataset("gary109/crop14_balance")
            # Use the provided splits; here, 'train' and 'validation' are available
            train_dataset_hf = dataset["train"]
            test_dataset_hf = dataset["validation"]
            num_classes = 14  # As given in the features ("14 classes") 
            input_size = dataset_stats["crop14_balance"]["input_size"]
            train_dataset = HFImageDataset(train_dataset_hf, transform=None, input_size=input_size)
            test_dataset = HFImageDataset(test_dataset_hf, transform=None, input_size=input_size)
            print(f"crop14_balance loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")
        except Exception as e:
            raise ValueError(f"Error loading crop14_balance dataset: {e}")


    else:
        raise ValueError(f"Dataset {dataset_name} not supported or MedMNIST not installed")
    
    
    # Get stats from our predefined dictionary
    stats = dataset_stats.get(dataset_name.lower(), {
        "mean": (0.5,),
        "std": (0.5,),
        "input_size": 28,
        "is_rgb": False
    })
    

    return train_dataset, test_dataset, num_classes, stats["input_size"], stats["mean"], stats["std"], stats["is_rgb"]

def calculate_dataset_stats(dataset, batch_size=64, max_samples=10000):
    """Calculate mean and std for dataset
    
    Args:
        dataset: PyTorch dataset or HuggingFace dataset
        batch_size: Batch size for loading
        max_samples: Maximum number of samples to use (for large datasets)
    
    Returns:
        mean, std as lists
    """
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Limit samples for large datasets
    if hasattr(dataset, '__len__') and len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset_subset = Subset(dataset, indices)
    else:
        dataset_subset = dataset
    
    # Create a copy of the dataset with only ToTensor transform
    if hasattr(dataset, 'transform'):
        # Standard PyTorch dataset
        original_transform = dataset.transform
        dataset.transform = torchvision.transforms.ToTensor()
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform'):
        # Handle subset case
        original_transform = dataset.dataset.transform
        dataset.dataset.transform = torchvision.transforms.ToTensor()
    else:
        # Create a wrapper for HuggingFace datasets or other types
        class StatsDataset(torch.utils.data.Dataset):
            def __init__(self, original_dataset):
                self.dataset = original_dataset
                self.transform = torchvision.transforms.ToTensor()
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                if hasattr(self.dataset, '__getitem__'):
                    item = self.dataset[idx]
                    if isinstance(item, tuple) and len(item) >= 2:
                        img, label = item[0], item[1]
                    else:
                        # For HuggingFace datasets
                        img = item['image']
                        label = item.get('label', 0)
                else:
                    # Fallback for unusual dataset structures
                    raise ValueError("Dataset structure not supported for statistics calculation")
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label
        
        dataset_subset = StatsDataset(dataset_subset)
        original_transform = None
    
    # Create loader
    loader = DataLoader(
        dataset_subset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    print("Calculating dataset statistics...")
    for data, _ in tqdm(loader):
        # Check if data is already a tensor
        if not isinstance(data, torch.Tensor):
            print(f"Warning: Expected tensor but got {type(data)}. Skipping batch.")
            continue
            
        # Handle both single-channel and multi-channel images
        if data.dim() == 3:  # [batch, height, width]
            data = data.unsqueeze(1)  # Add channel dimension
        
        # Mean over batch, height and width, but not over channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    # Restore original transform
    if hasattr(dataset, 'transform') and original_transform is not None:
        dataset.transform = original_transform
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'transform') and original_transform is not None:
        dataset.dataset.transform = original_transform
    
    # Calculate mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    
    return mean.tolist(), std.tolist()

def create_transforms(mean, std, input_size, is_rgb=True, da_strength=1):
    """Create training and testing transforms based on statistics and data augmentation strength"""
    
    # Always use 224x224 as the standard size
    standard_size = 224
    
    # Create base transform list starting with RGB conversion
    base_transforms = [
        # First step: Convert to RGB using proper torchvision v2 transform
        v2.RGB(),
        # Resize to standard 224x224 size
        transforms.Resize((standard_size, standard_size), antialias=True),
    ]
    
    # Basic test transform - RGB conversion, resize, normalize
    test_transform = transforms.Compose(base_transforms + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Create training transform with augmentations based on strength
    if da_strength > 0:
        aug_list = base_transforms + [
            transforms.RandomResizedCrop(standard_size, antialias=True),
            transforms.RandomHorizontalFlip(),
        ]
        
        if is_rgb and da_strength > 1:
            aug_list.extend([
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
                ], p=0.3),
                transforms.RandomGrayscale(p=0.2),
            ])
        
        if da_strength > 2 and is_rgb:
            aug_list.append(transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))
            ], p=0.2))
        
        aug_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        if da_strength > 2:
            aug_list.append(transforms.RandomErasing(p=0.25))
        
        train_transform = transforms.Compose(aug_list)
    else:
        train_transform = test_transform
    
    return train_transform, test_transform

def prepare_data_loaders(
    dataset_name, 
    batch_size, 
    num_diet_classes,
    da_strength=1, 
    limit_data=np.inf, 
    root='./data'
):
    """Prepare data loaders for DIET finetuning"""
    print(f"Loading {dataset_name} dataset...")
    
    # Get dataset
    training_data_raw, test_data_raw, num_classes, input_size, mean, std, is_rgb = get_dataset(dataset_name, root)
    print(f"Dataset loaded: input_size={input_size}, mean={mean}, std={std}, is_rgb={is_rgb}")
    
    # Create transforms
    train_transform, test_transform = create_transforms(mean, std, input_size, is_rgb, da_strength)
    
    # Apply transforms to the datasets
    if dataset_name.lower() in ["plantnet300k", "galaxy10_decals"]:
        # For HuggingFace datasets we need to handle the custom dataset wrapper
        training_data = training_data_raw  # No deepcopy needed
        test_data = test_data_raw
        if hasattr(training_data, 'transform'):
            training_data.transform = train_transform
            test_data.transform = test_transform
        else:
            print(f"Note: {dataset_name} dataset structure is using custom transform handling")
    else:
        # Standard datasets
        import copy
        training_data = copy.deepcopy(training_data_raw)
        try:
            training_data.transform = train_transform
            test_data = copy.deepcopy(test_data_raw)
            test_data.transform = test_transform
        except AttributeError:
            # Handle if dataset doesn't have a transform attribute (like Subset)
            print("Note: Using dataset that requires special transform handling")
    
    # Limit training data if specified
    if limit_data < np.inf and limit_data < len(training_data):
        print(f"Limiting training data to {limit_data} samples (out of {len(training_data)})")
        indices = torch.randperm(len(training_data))[:limit_data]
        training_data = Subset(training_data, indices)
    else:
        print(f"Using full training set: {len(training_data)} samples")
    
    # Special handling for Galaxy dataset
    if dataset_name.lower() == "galaxy10_decals":
        print("\n===== REBUILDING GALAXY DATASET FROM SCRATCH =====")
        
        # Get the raw dataset again
        from datasets import load_dataset
        
        raw_dataset = load_dataset("matthieulel/galaxy10_decals")
        train_data = raw_dataset["train"]
        test_data = raw_dataset["test"]
        
        # Create robust galaxy dataset
        print("Creating robust galaxy dataset...")
        training_data = RobustGalaxyDataset(
            train_data, 
            transform=train_transform,
            diet_classes=num_diet_classes,
            limit_samples=limit_data if limit_data < np.inf else None
        )
        
        test_data = RobustGalaxyDataset(
            test_data,
            transform=test_transform,
            diet_classes=num_diet_classes
        )
        
        print(f"Created robust datasets: {len(training_data)} training, {len(test_data)} test")
        print("===== GALAXY DATASET REBUILDING COMPLETE =====\n")
    else:
        # For non-Galaxy datasets, use the regular DatasetWithIndices wrapper
        training_data = DatasetWithIndices(training_data, num_diet_classes=num_diet_classes)
    
    print(f"Test set size: {len(test_data)} samples")
    
    # Create data loaders
    training_loader = DataLoader(
        training_data, 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=0
    )
    
    dataset_info = {
        'num_classes': num_classes,
        'input_size': input_size,
        'mean': mean,
        'std': std,
        'is_rgb': is_rgb,
        'train_size': len(training_data),
        'test_size': len(test_data)
    }
    
    return training_loader, test_loader, dataset_info