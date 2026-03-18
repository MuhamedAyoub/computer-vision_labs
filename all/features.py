"""Feature extraction classes"""
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor
from config import ExperimentConfig





class FeatureExtractor:
    """Extracts features for BoW: SIFT (local) or CNN conv-layer local descriptors."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        method = config.dsc_method.lower()
        if method == "sift":
            self.sift = cv2.SIFT_create(
                nfeatures=config.sift_n_features,
                contrastThreshold=getattr(config, "sift_contrast_threshold", 0.04)
            )
        else:
            self.sift = None

        # Defaults (can be set in config)
        self.img_size = getattr(config, "img_size", 224)
        self.batch_size = getattr(config, "cnn_batch_size", 32)
        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.max_total_descriptors = getattr(config, "max_total_descriptors", 20000000)
        self.l2_normalize = getattr(config, "l2_normalize", True)
        self.cnn_model = getattr(config, "cnn_model", "vgg16").lower()

        # build CNN extractor (lazy load)
        self._cnn_extractor = None
        self._cnn_out_channels = None

    def deep_nn_feature_extraction(self, images: List[np.ndarray], model_name: Optional[str] = None) -> np.ndarray:
        """
        Extract features from CNN models by removing the FC layer.
        Returns global features (not local descriptors for BoVW).
        
        Args:
            images: List of BGR images
            model_name: CNN model name ('resnet18', 'resnet50', 'vgg16', etc.)
        
        Returns:
            features_array: (N, feature_dim) array of global features
        """
        if model_name is None:
            model_name = self.cnn_model
        
        model_name = model_name.lower()
        
        # Load model and remove FC layer (classifier)
        if model_name == "vgg16":
            try:
                model = models.vgg16(pretrained=True)
            except Exception:
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            # Remove classifier, keep only features
            feature_extractor = nn.Sequential(*list(model.features.children())).to(self.device)
            # Add adaptive pooling to get fixed-size features
            feature_extractor = nn.Sequential(
                feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            feature_dim = 512
            
        elif model_name == "resnet18":
            try:
                model = models.resnet18(pretrained=True)
            except Exception:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Remove FC layer, keep everything else
            feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            feature_dim = 512
            
        elif model_name == "resnet34":
            try:
                model = models.resnet34(pretrained=True)
            except Exception:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            feature_dim = 512
            
        elif model_name == "resnet50":
            try:
                model = models.resnet50(pretrained=True)
            except Exception:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            feature_dim = 2048
            
        elif model_name == "resnet101":
            try:
                model = models.resnet101(pretrained=True)
            except Exception:
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            feature_dim = 2048
            
        else:
            raise ValueError(f"Unsupported model for direct feature extraction: {model_name}")
        
        # Freeze parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        
        # Process images in batches
        features = []
        n_images = len(images)
        
        for start in range(0, n_images, self.batch_size):
            end = min(n_images, start + self.batch_size)
            batch_imgs = images[start:end]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_imgs:
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Normalize
                img_norm = (img_rgb - self.normalize_mean[None, None, :]) / self.normalize_std[None, None, :]
                # to tensor CHW
                t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
                batch_tensors.append(t)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                output = feature_extractor(batch_tensor)  # (B, C, H, W) or (B, C, 1, 1)
                # Flatten to (B, feature_dim)
                output = output.view(output.size(0), -1).cpu().numpy()
                features.append(output)
            
            if end % max(100, self.batch_size) == 0 or end == n_images:
                print(f"  Processed {end}/{n_images} images")
        
        features_array = np.vstack(features)
        print(f"Extracted CNN features shape: {features_array.shape}")
        return features_array
    
    def vit_feature_extraction(self, images: List[np.ndarray], model_name: str = "google/vit-base-patch16-224") -> np.ndarray:
        """
        Extract features from Vision Transformer (ViT) by removing the classification head.
        
        Args:
            images: List of BGR images
            model_name: HuggingFace model name for ViT
        
        Returns:
            features_array: (N, feature_dim) array of global features
        """
        print(f"Loading ViT model: {model_name}")
        
        # Load ViT model and processor
        try:
            model = ViTModel.from_pretrained(model_name)
            processor = ViTImageProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model: {e}")
        
        model.to(self.device)
        model.eval()
        
        # Process images in batches
        features = []
        n_images = len(images)
        
        for start in range(0, n_images, self.batch_size):
            end = min(n_images, start + self.batch_size)
            batch_imgs = images[start:end]
            
            # Convert BGR to RGB and prepare for ViT
            batch_rgb = []
            for img in batch_imgs:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_rgb.append(img_rgb)
            
            # Process with ViT processor
            inputs = processor(images=batch_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding (first token) as global feature
                cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (B, hidden_dim)
                features.append(cls_features)
            
            if end % max(100, self.batch_size) == 0 or end == n_images:
                print(f"  Processed {end}/{n_images} images")
        
        features_array = np.vstack(features)
        print(f"Extracted ViT features shape: {features_array.shape}")
        return features_array
    # -----------------------------
    # Utility: build the CNN up to the chosen conv layer
    # -----------------------------
    def _build_cnn_extractor(self):
        if self._cnn_extractor is not None:
            return self._cnn_extractor, self._cnn_out_channels

        name = self.cnn_model

        # -------------------------------
        # 1. VGG16 (take last convolutional block)
        # -------------------------------
        if name == "vgg16":
            try:
                model = models.vgg16(pretrained=True)
            except Exception:
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

            # take features up to conv5_3 (index 30, exclusive)
            extractor = nn.Sequential(*list(model.features.children())[:30]).to(self.device)
            out_ch = 512  # conv5_3 has 512 channels

        # -------------------------------
        # 2. ResNet variants (RESOLVED: use deeper layers)
        # -------------------------------
        elif name in ("resnet18", "resnet34", "resnet50", "resnet101"):

            # --- Load model
            if name == "resnet18":
                try:
                    model = models.resnet18(pretrained=True)
                except Exception:
                    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif name == "resnet34":
                try:
                    model = models.resnet34(pretrained=True)
                except Exception:
                    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            elif name == "resnet50":
                try:
                    model = models.resnet50(pretrained=True)
                except Exception:
                    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                try:
                    model = models.resnet101(pretrained=True)
                except Exception:
                    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

            # ---------------------------------
            # FIX: Use deeper layers (layer3 + layer4)
            # ---------------------------------
            layers = [
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,   
                model.layer4   
            ]

            extractor = nn.Sequential(*layers).to(self.device)

            # output channel size depends on model depth
            if name in ("resnet18", "resnet34"):
                out_ch = 512     # layer4 output channels
            else:
                out_ch = 2048    # resnet50/resnet101 (bottleneck)

        else:
            raise ValueError(f"Unsupported cnn_model: {name}")

        for p in extractor.parameters():
            p.requires_grad = False
        extractor.eval()

        self._cnn_extractor = extractor
        self._cnn_out_channels = out_ch
        return extractor, out_ch

    # -----------------------------
    # CNN conv-layer descriptor extraction (batching)
    # -----------------------------
    def _cnn_conv_descriptors(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        For each image returns a (N_k x C) array where N_k = H*W spatial locations of chosen conv layer.
        Returns:
            image_descriptors: List[np.ndarray], each (N_k, C)
            all_descriptors_stacked: np.ndarray (M, C)
        """
        extractor, out_ch = self._build_cnn_extractor()
        n_images = len(images)

        image_descriptors: List[np.ndarray] = []
        all_desc_list: List[np.ndarray] = []

        # Process in batches
        for start in range(0, n_images, self.batch_size):
            end = min(n_images, start + self.batch_size)
            batch_imgs = images[start:end]

            # preprocess & stack
            batch_tensors = []
            for img in batch_imgs:
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Normalize
                img_norm = (img_rgb - self.normalize_mean[None, None, :]) / self.normalize_std[None, None, :]
                # to tensor CHW
                t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                batch_tensors.append(t)
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)  # (B, C, H, W)

            with torch.no_grad():
                feat = extractor(batch_tensor)  # (B, C_f, H_f, W_f)
                B, C_f, H_f, W_f = feat.shape
                # reshape to (B, H_f*W_f, C_f)
                feat = feat.permute(0, 2, 3, 1).reshape(B, H_f * W_f, C_f).cpu().numpy()

            # collect per-image
            for i in range(B):
                descriptors = feat[i]  # (H_f*W_f, C_f)
                if self.l2_normalize:
                    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    descriptors = descriptors / norms
                image_descriptors.append(descriptors.astype(np.float32))
                all_desc_list.append(descriptors.astype(np.float32))

            # optional progress print
            processed = end
            if processed % max(50, self.batch_size) == 0 or processed == n_images:
                print(f"  Processed {processed}/{n_images} images -> feature map {H_f}x{W_f}, channels={C_f}")

        # stack all and optionally subsample to limit total descriptors
        all_descriptors_stacked = np.vstack(all_desc_list).astype(np.float32)
        total = all_descriptors_stacked.shape[0]
        if total > self.max_total_descriptors:
            # random subsample to limit memory for kmeans
            idx = np.random.choice(total, self.max_total_descriptors, replace=False)
            all_descriptors_stacked = all_descriptors_stacked[idx]
            print(f"  Subsampled descriptors: {self.max_total_descriptors}/{total}")

        print(f"Total CNN local descriptors (stacked): {all_descriptors_stacked.shape}")
        return image_descriptors, all_descriptors_stacked

 
    def _sift_descriptors(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        image_descriptors = []
        all_desc = []
        for i, img in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(images)} images")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is None:
                descriptors = np.zeros((0, 128), dtype=np.float32)
            image_descriptors.append(descriptors.astype(np.float32))
            if descriptors.shape[0] > 0:
                all_desc.append(descriptors.astype(np.float32))
        if len(all_desc) == 0:
            raise RuntimeError("No SIFT descriptors found in any image")
        all_descriptors_stacked = np.vstack(all_desc).astype(np.float32)
        print(f"Total SIFT descriptors: {all_descriptors_stacked.shape}")
        return image_descriptors, all_descriptors_stacked
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Public API: choose method based on config
    # -----------------------------
    def extract_descriptors(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        method = self.config.dsc_method.lower()
        print(f"Descriptor method: {method.upper()}")

        if method == "sift":
            return self._sift_descriptors(images)
        elif method == "cnn":
            return self._cnn_conv_descriptors(images)
        else:
            raise ValueError(f"Unknown descriptor method '{self.config.dsc_method}'")
