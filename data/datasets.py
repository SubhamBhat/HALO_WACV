import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import math


class HAM10000Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_transforms(allow_vflip=False):
    """Defines image augmentations based on ImageNet standards."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tf_list = [
        transforms.Resize((cfg.IMAGE_SIZE + 32, cfg.IMAGE_SIZE + 32)),
        transforms.RandomCrop(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
    ]
    if allow_vflip:
        train_tf_list.append(transforms.RandomVerticalFlip())
        
    train_tf_list.extend([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_tf = transforms.Compose(train_tf_list)
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf

def setup_ham10000():
    """Finds HAM10000 images on Kaggle, splits data, and builds DataLoaders."""
    print("Setting up HAM10000 data...")
    
    meta_path = None
    for pat in ["**/HAM10000_metadata*", "**/hmnist_metadata*"]:
        candidates = glob.glob(os.path.join(cfg.KAGGLE_INPUT, pat), recursive=True)
        for p in candidates:
            if "image_id" in pd.read_csv(p, nrows=1).columns:
                meta_path = p
                break
        if meta_path: break
    if not meta_path: raise FileNotFoundError("HAM10000 metadata not found!")
    
    print(f"  Metadata: {meta_path}")
    df = pd.read_csv(meta_path)
    print(f"  Metadata rows: {len(df)}")
    
    # Search for images ONLY in the same dataset folder (not all of /kaggle/input)
    dataset_root = os.path.dirname(meta_path)
    image_map = {}
    for root, dirs, files in os.walk(dataset_root):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_map[os.path.splitext(fname)[0]] = os.path.join(root, fname)
    print(f"  Images found: {len(image_map)}")
            
    df["image_path"] = df["image_id"].map(image_map)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df["label"] = df["dx"].map(cfg.CLASS_LABELS)
    print(f"  Matched rows: {len(df)}")
    
    cfg.CLASS_COUNTS = [df[df["label"] == i].shape[0] for i in range(cfg.NUM_CLASSES)]
    print(f"  Class counts: {dict(zip(cfg.CLASS_NAMES, cfg.CLASS_COUNTS))}")
    
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    
    # Proportional reduction of training set to speed up experiments
    if hasattr(cfg, 'HAM_REDUCTION_FACTOR') and cfg.HAM_REDUCTION_FACTOR < 1.0:
        reduced_dfs = []
        for label_val in range(cfg.NUM_CLASSES):
            class_df = train_df[train_df["label"] == label_val]
            n_keep = max(1, int(len(class_df) * cfg.HAM_REDUCTION_FACTOR))
            reduced_dfs.append(class_df.sample(n=n_keep, random_state=42))
        train_df = pd.concat(reduced_dfs).reset_index(drop=True)
        print(f"  HAM10000 reduced to {len(train_df)} training samples ({cfg.HAM_REDUCTION_FACTOR*100:.0f}%)")
    
    train_df, val_df = train_test_split(train_df, test_size=0.15/0.85, stratify=train_df["label"], random_state=42)
    print(f"  Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    train_tf, eval_tf = get_transforms(allow_vflip=True)
    mk_dl = lambda d, tf, shuf: DataLoader(
        HAM10000Dataset(d["image_path"].tolist(), d["label"].tolist(), tf),
        batch_size=cfg.BATCH_SIZE, shuffle=shuf, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    print("  DataLoaders ready.\n")
    return mk_dl(train_df, train_tf, True), mk_dl(val_df, eval_tf, False), mk_dl(test_df, eval_tf, False)

def setup_cifar10_lt():
    """Creates an artificially imbalanced CIFAR-10 dataset (50:1 ratio)."""
    print("Setting up CIFAR-10 Long-Tail data...")
    
    cfg.NUM_CLASSES = 10
    cfg.CLASS_NAMES = cfg.CIFAR10_CLASS_NAMES
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    # AlexNet, VGG, and GoogLeNet physically crash on 32x32 due to their rigid pooling/kernel sizes.
    if cfg.BACKBONE in ["alexnet", "vgg16", "googlenet"]:
        cfg.IMAGE_SIZE = 224
    else:
        cfg.IMAGE_SIZE = 32
        
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    
    # If 32x32, padding=4. If 224x224, padding=28
    pad_size = 4 if cfg.IMAGE_SIZE == 32 else 28
    
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomCrop(cfg.IMAGE_SIZE, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Try loading from Kaggle dataset first (no internet needed)
    # Add "cifar-10-python" or "cifar10" dataset in Kaggle to use this
    cifar_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/data_batch_1"), recursive=True):
        if os.path.isfile(candidate):
            cifar_local = os.path.dirname(candidate)
            break
    
    root = "/tmp/cifar10_data"
    dl = True
    if cifar_local:
        print(f"  Loading CIFAR-10 from local: {cifar_local}")
        dl = False
        target_dir = os.path.join(root, "cifar-10-batches-py")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(cifar_local):
                src = os.path.join(cifar_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local CIFAR-10 found, downloading (enable Internet in Kaggle Settings)...")
    
    full_train = tv_datasets.CIFAR10(root=root, train=True, download=dl)
    full_test = tv_datasets.CIFAR10(root=root, train=False, download=dl)
    
    targets = np.array(full_train.targets)
    selected_indices = []
    for c in range(10):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = cfg.CIFAR10_SAMPLES_PER_CLASS[c]
        chosen = np.random.choice(class_indices, size=min(n_keep, len(class_indices)), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets[train_idx]
    cfg.CLASS_COUNTS = [int((train_targets == c).sum()) for c in range(10)]
    print(f"  Long-Tail Class counts (train): {dict(zip(cfg.CLASS_NAMES, cfg.CLASS_COUNTS))}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = TransformSubset(full_test, list(range(len(full_test))), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

def setup_cifar100_lt():
    """Creates an artificially imbalanced CIFAR-100 dataset."""
    print("Setting up CIFAR-100 Long-Tail data...")
    
    cfg.NUM_CLASSES = 100
    cfg.CLASS_NAMES = [f"class_{i}" for i in range(100)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    if cfg.BACKBONE in ["alexnet", "vgg16", "googlenet"]:
        cfg.IMAGE_SIZE = 224
    else:
        cfg.IMAGE_SIZE = 32
    
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    pad_size = 4 if cfg.IMAGE_SIZE == 32 else 28
    
    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomCrop(cfg.IMAGE_SIZE, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Search locally first, then download
    cifar_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/meta"), recursive=True):
        if "cifar100" in candidate.lower() or "cifar-100" in candidate.lower():
            cifar_local = os.path.dirname(candidate)
            break
            
    # Check original torchvision format just in case
    if not cifar_local:
        for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/cifar-100-python"), recursive=True):
            if os.path.isdir(candidate):
                cifar_local = os.path.dirname(candidate)
                break
    
    root = "/tmp/cifar100_data"
    dl = True
    if cifar_local:
        print(f"  Loading CIFAR-100 from local: {cifar_local}")
        dl = False
        # Torchvision strictly expects the 'cifar-100-python' directory name inside the root
        target_dir = os.path.join(root, "cifar-100-python")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(cifar_local):
                src = os.path.join(cifar_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local CIFAR-100 found, downloading...")
    
    full_train = tv_datasets.CIFAR100(root=root, train=True, download=dl)
    full_test = tv_datasets.CIFAR100(root=root, train=False, download=dl)
    
    # Create exponential decay long-tail distribution
    targets = np.array(full_train.targets)
    selected_indices = []
    imb_factor = cfg.CIFAR100_IMB_FACTOR
    max_num = cfg.CIFAR100_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_targets == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = TransformSubset(full_test, list(range(len(full_test))), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

def setup_oxford_pet():
    """Loads Oxford-IIIT Pet dataset (37 breeds, natural imbalance)."""
    print("Setting up Oxford-IIIT Pet data...")
    
    cfg.NUM_CLASSES = 37
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"breed_{i}" for i in range(37)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # Search locally first
    pet_local = None
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/annotations"), recursive=True):
        if os.path.isdir(candidate):
            pet_local = os.path.dirname(candidate)
            break
    
    root = "/tmp/oxford_pet"
    dl = True
    if pet_local:
        print(f"  Loading Oxford Pet from local: {pet_local}")
        dl = False
        target_dir = os.path.join(root, "oxford-iiit-pet")
        os.makedirs(target_dir, exist_ok=True)
        try:
            for f in os.listdir(pet_local):
                src = os.path.join(pet_local, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except Exception as e:
            print(f"  Warning: Symlink failed, falling back to download. {e}")
            dl = True
    else:
        print("  No local Oxford Pet found, downloading...")
    
    full_train = tv_datasets.OxfordIIITPet(root=root, split="trainval", download=dl)
    full_test = tv_datasets.OxfordIIITPet(root=root, split="test", download=dl)
    
    # Create exponential decay long-tail distribution
    targets = np.array([t for _, t in full_train])
    selected_indices = []
    imb_factor = cfg.OXFORD_PET_IMB_FACTOR
    max_num = cfg.OXFORD_PET_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_targets = targets
    
    train_tgts = train_targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
    
    class FullDatasetTransform(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.transform: img = self.transform(img)
            return img, label
    
    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds = TransformSubset(full_train, val_idx, eval_tf)
    test_ds = FullDatasetTransform(full_test, eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

def setup_cub200():
    """Loads CUB-200-2011 fine-grained bird classification (200 classes)."""
    print("Setting up CUB-200-2011 data...")
    
    cfg.NUM_CLASSES = 200
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"bird_{i}" for i in range(200)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # Search for CUB dataset in Kaggle inputs
    cub_root = None
    is_imagefolder = False
    
    # 1. Search for ImageFolder structure (like muteks/cub-200-2011)
    for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/train"), recursive=True):
        if "cub" in candidate.lower() or "bird" in candidate.lower():
            parent = os.path.dirname(candidate)
            if os.path.exists(os.path.join(parent, "test")):
                cub_root = parent
                is_imagefolder = True
                break
                
    # 2. Search for original Caltech metadata format
    if cub_root is None:
        for candidate in glob.glob(os.path.join(cfg.KAGGLE_INPUT, "**/images.txt"), recursive=True):
            parent = os.path.dirname(candidate)
            if os.path.exists(os.path.join(parent, "image_class_labels.txt")):
                cub_root = parent
                break
    
    if cub_root is None:
        print("  No local CUB-200 found. Attempting download...")
        cub_root = "/tmp/cub200"
        os.makedirs(cub_root, exist_ok=True)
        import urllib.request, tarfile
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        tar_path = os.path.join(cub_root, "CUB_200_2011.tgz")
        if not os.path.exists(os.path.join(cub_root, "CUB_200_2011")):
            print("  Downloading CUB-200-2011...")
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(cub_root)
        cub_root = os.path.join(cub_root, "CUB_200_2011")
    
    print(f"  Loading CUB-200 from: {cub_root}")
    
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    if is_imagefolder:
        from torchvision import datasets
        train_ds_temp = datasets.ImageFolder(os.path.join(cub_root, "train"))
        test_ds_temp = datasets.ImageFolder(os.path.join(cub_root, "test"))
        for path, label in train_ds_temp.samples:
            train_paths.append(path)
            train_labels.append(label)
        for path, label in test_ds_temp.samples:
            test_paths.append(path)
            test_labels.append(label)
    else:
        # Parse CUB-200 metadata files
        with open(os.path.join(cub_root, "images.txt")) as f:
            id_to_path = {int(line.split()[0]): line.split()[1] for line in f.readlines()}
        with open(os.path.join(cub_root, "image_class_labels.txt")) as f:
            id_to_label = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f.readlines()}
        with open(os.path.join(cub_root, "train_test_split.txt")) as f:
            id_to_split = {int(line.split()[0]): int(line.split()[1]) for line in f.readlines()}
        
        img_dir = os.path.join(cub_root, "images")
        
        for img_id, rel_path in id_to_path.items():
            full_path = os.path.join(img_dir, rel_path)
            label = id_to_label[img_id]
            if id_to_split[img_id] == 1:
                train_paths.append(full_path)
                train_labels.append(label)
            else:
                test_paths.append(full_path)
                test_labels.append(label)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # Create exponential decay long-tail distribution
    selected_indices = []
    imb_factor = cfg.CUB200_IMB_FACTOR
    max_num = cfg.CUB200_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(train_labels == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = train_labels[selected_indices]
    
    # Split train into train/val
    train_idx, val_idx = train_test_split(selected_indices, test_size=0.15,
                                           stratify=sel_targets, random_state=42)
    
    train_tgts = train_labels[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")
    
    class SubsetByIndex(Dataset):
        def __init__(self, paths, labels, indices, transform):
            self.paths = [paths[i] for i in indices]
            self.labels = [labels[i] for i in indices]
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
    
    class ImageListDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
    
    train_ds = SubsetByIndex(train_paths, train_labels.tolist(), train_idx, train_tf)
    val_ds = SubsetByIndex(train_paths, train_labels.tolist(), val_idx, eval_tf)
    test_ds = ImageListDataset(test_paths, test_labels.tolist(), eval_tf)
    
    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

class StanfordCarsRawDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        import scipy.io
        import pandas as pd
        
        # Determine image dir
        img_dir_name = "cars_train" if split == "train" else "cars_test"
        img_dir = None
        
        # Deep search for the actual directory containing the JPGs (bypasses nested folders)
        for candidate in glob.glob(os.path.join(root_dir, f"**/{img_dir_name}/**/*.jpg"), recursive=True):
            img_dir = os.path.dirname(candidate)
            break
            
        if not img_dir:
            for candidate in glob.glob(os.path.join(root_dir, f"**/{img_dir_name}"), recursive=True):
                if os.path.isdir(candidate):
                    img_dir = candidate
                    break
        
        if img_dir is None:
            raise FileNotFoundError(f"Could not find {img_dir_name} directory in {root_dir}")
            
        # Try to find CSV annotations first
        csv_path = None
        for candidate in glob.glob(os.path.join(root_dir, f"**/anno_{split}.csv"), recursive=True):
            csv_path = candidate
            break
            
        if csv_path:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Kaggle CSVs sometimes just have the filename
                fname = row.iloc[0]
                # Label is usually 1-indexed in Stanford Cars
                label = int(row.iloc[-1]) - 1
                self.samples.append((os.path.join(img_dir, fname), label))
        else:
            # Fallback to .mat
            mat_path = None
            mat_name = "cars_train_annos.mat" if split == "train" else "cars_test_annos_withlabels.mat"
            for candidate in glob.glob(os.path.join(root_dir, f"**/{mat_name}"), recursive=True):
                mat_path = candidate
                break
                
            if not mat_path:
                # Some test mats are named cars_test_annos.mat (without labels). We need labels!
                for candidate in glob.glob(os.path.join(root_dir, f"**/cars_test_annos.mat"), recursive=True):
                    mat_path = candidate
                    break
                    
            if not mat_path:
                raise FileNotFoundError(f"Could not find annotations for {split}")
                
            annos = scipy.io.loadmat(mat_path)
            if 'annotations' in annos:
                for anno in annos['annotations'][0]:
                    fname = anno[-1][0]
                    # Check if label exists (test set might not have it in some versions)
                    if len(anno) >= 5:
                        label = int(anno[4][0][0]) - 1
                        self.samples.append((os.path.join(img_dir, fname), label))
                    else:
                        self.samples.append((os.path.join(img_dir, fname), 0))
            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

def setup_stanford_cars():
    """Loads Stanford Cars fine-grained classification (196 classes)."""
    print("Setting up Stanford Cars data...")
    
    cfg.NUM_CLASSES = 196
    cfg.IMAGE_SIZE = 224
    cfg.CLASS_NAMES = [f"car_{i}" for i in range(196)]
    cfg.CLASS_LABELS = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    
    train_tf, eval_tf = get_transforms()
    
    # 1. Check if user attached dataset to Kaggle
    kaggle_root = None
    if getattr(cfg, 'IS_KAGGLE', True) or os.path.exists("/kaggle/input"):
        base_dir = getattr(cfg, 'KAGGLE_INPUT', "/kaggle/input")
        # Look for imagefolder format first
        for candidate in glob.glob(os.path.join(base_dir, "**/stanford-cars/train"), recursive=True):
            kaggle_root = os.path.dirname(candidate)
            break
        # Look for raw format
        if not kaggle_root:
            for candidate in glob.glob(os.path.join(base_dir, "**/cars_train"), recursive=True):
                kaggle_root = os.path.dirname(os.path.dirname(candidate)) # Go up past cars_train/cars_train if nested
                if not os.path.exists(os.path.join(kaggle_root, "cars_train")):
                     kaggle_root = os.path.dirname(candidate)
                break

    full_train = None
    full_test = None
    
    if kaggle_root:
        print(f"  Found Stanford Cars dataset locally at: {kaggle_root}")
        if os.path.exists(os.path.join(kaggle_root, "train")):
            print("  Using ImageFolder format.")
            full_train = tv_datasets.ImageFolder(os.path.join(kaggle_root, "train"))
            full_test = tv_datasets.ImageFolder(os.path.join(kaggle_root, "test"))
        else:
            print("  Using Raw/CSV/MAT format.")
            full_train = StanfordCarsRawDataset(kaggle_root, split="train")
            full_test = StanfordCarsRawDataset(kaggle_root, split="test")
    else:
        # Fallback to downloading
        root = "/tmp/stanford_cars"
        target_dir = os.path.join(root, "stanford-cars")
        if not os.path.exists(os.path.join(target_dir, "train")):
            print("  Downloading Stanford Cars from Fast.ai AWS mirror...")
            os.makedirs(root, exist_ok=True)
            import urllib.request, tarfile
            url = "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"
            tar_path = os.path.join(root, "stanford-cars.tgz")
            if not os.path.exists(tar_path):
                urllib.request.urlretrieve(url, tar_path)
            print("  Extracting Stanford Cars...")
            with tarfile.open(tar_path) as tf:
                tf.extractall(root)
        
        full_train = tv_datasets.ImageFolder(os.path.join(target_dir, "train"))
        full_test = tv_datasets.ImageFolder(os.path.join(target_dir, "test"))
    
    # Create exponential decay long-tail distribution
    if hasattr(full_train, 'targets'):
        targets = np.array(full_train.targets)
    else:
        targets = np.array([s[1] for s in full_train.samples])
        
    selected_indices = []
    imb_factor = cfg.STANFORD_CARS_IMB_FACTOR
    max_num = cfg.STANFORD_CARS_MAX_SAMPLES
    samples_per_cls = [int(max_num * (imb_factor ** (c / (cfg.NUM_CLASSES - 1.0)))) for c in range(cfg.NUM_CLASSES)]
    
    for c in range(cfg.NUM_CLASSES):
        class_indices = np.where(targets == c)[0]
        np.random.seed(42 + c)
        n_keep = min(samples_per_cls[c], len(class_indices))
        chosen = np.random.choice(class_indices, size=max(2, n_keep), replace=False)
        selected_indices.extend(chosen.tolist())
    
    np.random.seed(42)
    np.random.shuffle(selected_indices)
    
    sel_targets = targets[selected_indices]
    # 3-way stratified split from labeled training data
    # (Official Stanford Cars test set is unlabeled in Kaggle snapshots → all labels = -1)
    train_val_idx, test_idx = train_test_split(
        selected_indices, test_size=0.15, stratify=sel_targets, random_state=42)
    tv_targets = targets[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15/0.85, stratify=tv_targets, random_state=42)

    train_tgts = targets[train_idx]
    cfg.CLASS_COUNTS = [max(1, int((train_tgts == c).sum())) for c in range(cfg.NUM_CLASSES)]
    print(f"  Total train samples: {len(train_idx)}")
    print(f"  Imbalance ratio: {max(cfg.CLASS_COUNTS)}/{min(cfg.CLASS_COUNTS)} = {max(cfg.CLASS_COUNTS)/max(1,min(cfg.CLASS_COUNTS)):.1f}:1")

    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label

    train_ds = TransformSubset(full_train, train_idx, train_tf)
    val_ds   = TransformSubset(full_train, val_idx,   eval_tf)
    test_ds  = TransformSubset(full_train, test_idx,  eval_tf)

    print(f"  Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    mk_dl = lambda ds, shuf: DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuf,
                                          num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print("  DataLoaders ready.\n")
    return mk_dl(train_ds, True), mk_dl(val_ds, False), mk_dl(test_ds, False)

def setup_data():
    dispatchers = {
        "cifar10_lt": setup_cifar10_lt,
        "cifar100_lt": setup_cifar100_lt,
        "oxford_pet": setup_oxford_pet,
        "cub200": setup_cub200,
        "stanford_cars": setup_stanford_cars,
        "ham10000": setup_ham10000,
    }
    if cfg.DATASET not in dispatchers:
        raise ValueError(f"Unknown dataset: {cfg.DATASET}. Choose from: {list(dispatchers.keys())}")
    return dispatchers[cfg.DATASET]()

