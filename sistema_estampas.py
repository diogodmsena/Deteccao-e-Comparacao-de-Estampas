"""
Sistema Avançado de Comparação de Estampas
Pipeline: YOLOv8/RFDETR (Detecção) → Siamese Network (Comparação) → K-Fold Cross-Validation
Otimizações: TorchScript JIT, Half Precision, Batch Processing
Data: 2025
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image, ImageDraw
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from datetime import datetime
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import json
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Importações para YOLOv8 (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_DISPONIVEL = True
except ImportError:
    YOLO_DISPONIVEL = False
    print("⚠️  Ultralytics não instalado. Use: pip install ultralytics")

# ==================== CONFIGURAÇÕES ====================
class Config:
    """Configurações carregadas de arquivo externo"""
    # Valores padrão (caso o json falhe)
    config_data = {}
    try:
        with open("config.json", "r") as f:
            config_data = json.load(f)
            print("✓ Configurações carregadas do config.json")
    except FileNotFoundError:
        print("⚠ config.json não encontrado, usando defaults do código.")

    # Diretórios (lendo do JSON ou usando default)
    DIR_REFERENCIA = config_data.get("DIR_REFERENCIA", "./imagens_referencia")
    DIR_VALIDACAO = config_data.get("DIR_VALIDACAO", "./imagens_validacao")
    DIR_CHECKPOINTS = config_data.get("DIR_CHECKPOINTS", "./checkpoints")
    DIR_YOLO_WEIGHTS = config_data.get("DIR_YOLO_WEIGHTS", "./models")
    
    # Modelos
    YOLO_MODEL = config_data.get("YOLO_MODEL", "estampa_yolov8n_best.pt")
    
    """Configurações centralizadas do sistema"""
    # Diretórios
    DIR_TREINAMENTO = "./dataset_treinamento"
    
    # Parâmetros de comparação
    THRESHOLD_SIMILARIDADE = config_data.get("THRESHOLD_SIMILARIDADE", 0.95)
    EMBEDDING_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Parâmetros de detecção YOLOv8
    YOLO_CONFIDENCE = 0.30
    YOLO_IOU_THRESHOLD = 0.55
    YOLO_IMGSZ = 640
    YOLO_SELECT_STRATEGY = "confidence"
    CLASSE_ESTAMPA = 0  # Ajustar conforme fine-tuning do YOLO

    # Treino YOLO (opcional, se usar fine_tune_yolo)
    YOLO_BATCH = 16
    YOLO_RUN_NAME = "estampas_detector"
    YOLO_HYP = "configs/hyp_estampas.yaml"      # se existir no seu projeto
    YOLO_CLOSE_MOSAIC = 10
    YOLO_PATIENCE = 30
    YOLO_COS_LR = True
    YOLO_OPTIMIZER = "sgd"
    YOLO_LR0 = 0.005
    YOLO_LRF = 0.1
    YOLO_AMP = True
    
    # Parâmetros de otimização de inferência
    USE_HALF_PRECISION = True  # FP16 para GPUs modernas
    USE_JIT_COMPILE = True     # TorchScript JIT compilation
    BATCH_INFERENCE = True     # Processar múltiplas imagens em batch
    MAX_BATCH_SIZE = 8
    
    # Parâmetros de Fine-tuning com K-Fold Cross-Validation
    BATCH_SIZE = 32  # Aumentado para melhor aproveitamento de GPU
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    TRIPLET_MARGIN = 0.3
    MINING_STRATEGY = "batch-all"
    SAVE_INTERVAL = 5
    
    # Parâmetros de K-Fold Cross-Validation
    K_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 10
    
    # Cache de embeddings para referências
    CACHE_EMBEDDINGS = True
    CACHE_FILE = "./checkpoints/embeddings_cache.pt"

# ==================== DETECTOR YOLOV8 ====================
class DetectorEstampaYOLO:
    """
    Detector de estampas usando YOLOv8.
    Otimizado para inferência em tempo real.
    Mantém a interface pública original e acrescenta:
      - Estratégia de seleção de caixa ("confidence" ou "area") via config
      - Parametrização explícita de conf/iou/imgsz a partir do objeto Config
      - Robustez no tratamento de coordenadas e ausência de detecções
    """

    def __init__(self, config: "Config"):
        self.config = config

        if not YOLO_DISPONIVEL:
            raise ImportError("YOLOv8 não disponível. Instale: pip install ultralytics")

        # Caminho do modelo (peso treinado)
        model_path = Path(config.DIR_YOLO_WEIGHTS) / config.YOLO_MODEL
        if not model_path.exists():
            print(f"📥 Baixando modelo YOLOv8: {config.YOLO_MODEL}")
            model_path = config.YOLO_MODEL  # Ultralytics baixa automaticamente se for alias oficial

        # Carrega o modelo
        self.model = YOLO(str(model_path))

        # Força device explícito em string para Ultralytics
        self._device_str = None
        if getattr(self.config, "DEVICE", None) is not None:
            try:
                # torch.device('cuda:0') -> "0"; CPU -> "cpu"
                if self.config.DEVICE.type == "cuda":
                    self._device_str = "0"  # ajuste se usar múltiplas GPUs
                else:
                    self._device_str = "cpu"
            except Exception:
                self._device_str = None

        # Warmup: uma imagem dummy (mais estabilidade de latência)
        try:
            import numpy as np
            dummy = np.zeros((self.config.YOLO_IMGSZ, self.config.YOLO_IMGSZ, 3), dtype=np.uint8)
            _ = self.model.predict(source=dummy,
                                   conf=self.config.YOLO_CONFIDENCE,
                                   iou=self.config.YOLO_IOU_THRESHOLD,
                                   imgsz=self.config.YOLO_IMGSZ,
                                   verbose=False,
                                   half=getattr(self.config, "USE_HALF_PRECISION", False),
                                   stream=False,
                                   device=self._device_str)
        except Exception:
            pass

        # Estratégia de seleção de caixa: "confidence" (padrão) ou "area"
        self._select_strategy = getattr(self.config, "YOLO_SELECT_STRATEGY", "confidence")
        if self._select_strategy not in ("confidence", "area"):
            self._select_strategy = "confidence"

        # Otimizações
        # Half Precision (FP16) somente em CUDA
        if getattr(self.config, "USE_HALF_PRECISION", False) and getattr(self.config, "DEVICE", None) is not None:
            try:
                if self.config.DEVICE.type == "cuda":
                    # Em alguns builds, .model.half() é suportado; Ultralytics também aceita half no predict()
                    self.model.model.half()
                    print("✓ Half Precision (FP16) ativado para YOLOv8")
            except Exception as e:
                # Não interrompe o fluxo se não suportado
                print(f"⚠ Não foi possível ativar FP16 diretamente no modelo: {e}")

        print(f"✓ YOLOv8 carregado: {config.YOLO_MODEL}")

    def _choose_box_idx(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray
    ) -> int:
        """
        Seleciona o índice da caixa preferida conforme self._select_strategy.
        - "confidence": maior confiança
        - "area": maior área (w*h)
        Retorna -1 se não houver caixas.
        """
        if boxes_xyxy is None or boxes_xyxy.shape[0] == 0:
            return -1

        if self._select_strategy == "area":
            x1 = boxes_xyxy[:, 0]
            y1 = boxes_xyxy[:, 1]
            x2 = boxes_xyxy[:, 2]
            y2 = boxes_xyxy[:, 3]
            areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            return int(np.argmax(areas))

        # Padrão: maior confiança
        return int(np.argmax(confidences))

    def _predict_internal(self, source, stream: bool = False):
        """
        Invólucro de predict para centralizar parâmetros.
        source: caminho de imagem, lista de caminhos ou np.ndarray (BGR/RGB aceito pela Ultralytics)
        stream: True para batches grandes, economiza memória
        """
        return self.model.predict(
            source=source,
            conf=self.config.YOLO_CONFIDENCE,
            iou=self.config.YOLO_IOU_THRESHOLD,
            imgsz=self.config.YOLO_IMGSZ,
            verbose=False,
            half=getattr(self.config, "USE_HALF_PRECISION", False),
            stream=stream,
            device=self._device_str
        )

    def detectar_estampa(self, imagem_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta a região da estampa usando YOLOv8.
        Retorna: (x1, y1, x2, y2) ou None
        """
        try:
            results = self._predict_internal(imagem_path, stream=False)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            # boxes.xyxy: (N, 4), boxes.conf: (N,)
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            # Seleciona caixa final conforme estratégia
            best_idx = self._choose_box_idx(xyxy, confidences)
            if best_idx < 0:
                return None

            bbox = xyxy[best_idx].astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamping defensivo (coordenadas não negativas, x2>x1, y2>y1)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            return (x1, y1, x2, y2)

        except Exception as e:
            print(f"✗ Erro na detecção YOLOv8: {e}")
            return None

    def detectar_batch(self, imagens_paths: List[str]) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Detecta estampas em múltiplas imagens (batch processing).
        Muito mais eficiente que processar uma por uma.
        Retorna lista de bboxes ou None por imagem na mesma ordem da entrada.
        """
        try:
            results_iter = self._predict_internal(imagens_paths, stream=True)

            bboxes: List[Optional[Tuple[int, int, int, int]]] = []
            for result in results_iter:
                if result.boxes is None or len(result.boxes) == 0:
                    bboxes.append(None)
                    continue

                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                best_idx = self._choose_box_idx(xyxy, confidences)
                if best_idx < 0:
                    bboxes.append(None)
                    continue

                bbox = xyxy[best_idx].astype(int)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Clamping defensivo
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

                bboxes.append((x1, y1, x2, y2))

            return bboxes

        except Exception as e:
            print(f"✗ Erro na detecção batch: {e}")
            return [None] * len(imagens_paths)

    def extrair_estampa(self, imagem_path: str, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """
        Extrai a região da estampa a partir do bounding box.
        Retorna PIL.Image ou None em caso de falha.
        """
        try:
            img = Image.open(imagem_path).convert("RGB")
            x1, y1, x2, y2 = bbox

            # Boundaries defensivos
            w, h = img.size
            x1 = int(max(0, min(x1, w - 1)))
            y1 = int(max(0, min(y1, h - 1)))
            x2 = int(max(1, min(x2, w)))
            y2 = int(max(1, min(y2, h)))
            if x2 <= x1 or y2 <= y1:
                return None

            estampa = img.crop((x1, y1, x2, y2))
            return estampa

        except Exception as e:
            print(f"✗ Erro ao extrair estampa: {e}")
            return None

    def fine_tune_yolo(self, dataset_path: str, epochs: int = 50):
        """
        Fine-tuning do YOLOv8 para detecção específica de estampas.
        Dataset deve estar no formato YOLO (data.yaml ou caminho com data.yaml).
        Usa parâmetros de treino da própria Config quando disponíveis.
        """
        print(f"\n{'=' * 70}")
        print("🎯 FINE-TUNING YOLOv8 PARA DETECÇÃO DE ESTAMPAS")
        print(f"{'=' * 70}\n")

        # Parâmetros base vindos de Config, com defaults seguros
        imgsz = getattr(self.config, "YOLO_IMGSZ", 640)
        batch = getattr(self.config, "YOLO_BATCH", 16)
        device = getattr(self.config, "DEVICE", 0)
        project = getattr(self.config, "DIR_YOLO_WEIGHTS", "./runs")
        name = getattr(self.config, "YOLO_RUN_NAME", "estampas_detector")
        hyp = getattr(self.config, "YOLO_HYP", None)  # ex.: "configs/hyp_estampas.yaml"
        close_mosaic = getattr(self.config, "YOLO_CLOSE_MOSAIC", 10)
        patience = getattr(self.config, "YOLO_PATIENCE", 30)
        cos_lr = getattr(self.config, "YOLO_COS_LR", True)
        optimizer = getattr(self.config, "YOLO_OPTIMIZER", "sgd")
        lr0 = getattr(self.config, "YOLO_LR0", 0.005)
        lrf = getattr(self.config, "YOLO_LRF", 0.1)
        amp = getattr(self.config, "YOLO_AMP", True)

        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            hyp=hyp,
            close_mosaic=close_mosaic,
            patience=patience,
            cos_lr=cos_lr,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            amp=amp
        )

        print("\n✓ Fine-tuning concluído!")
        return results
    
    #inferir diretamente a partir de um array BGR, evitando que a Ultralytics reabra o arquivo do disco
    def detectar_estampa_array(self, image_bgr: "np.ndarray") -> Optional[Tuple[int, int, int, int]]:
        """
        Versão de detecção que recebe a imagem já carregada (BGR, OpenCV).
        Evita overhead de I/O no Ultralytics ao passar caminhos de arquivo.
        Retorna (x1, y1, x2, y2) ou None.
        """
        try:
            results = self._predict_internal(image_bgr, stream=False)
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            best_idx = self._choose_box_idx(xyxy, confidences)
            if best_idx < 0:
                return None

            bbox = xyxy[best_idx].astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamping defensivo
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            return (x1, y1, x2, y2)

        except Exception as e:
            print(f"✗ Erro na detecção YOLOv8 (array): {e}")
            return None
        
    #Ganho: elimina leitura dupla e conversão PIL, reduzindo centenas de milissegundos a segundos, conforme disco/FS.
    @staticmethod
    def extrair_estampa_array(image_bgr: "np.ndarray", bbox: Tuple[int, int, int, int]) -> Optional["np.ndarray"]:
        """
        Extrai recorte BGR direto via NumPy, sem PIL.
        Retorna np.ndarray BGR ou None.
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image_bgr.shape[:2]
            x1 = int(max(0, min(x1, w - 1)))
            y1 = int(max(0, min(y1, h - 1)))
            x2 = int(max(1, min(x2, w)))
            y2 = int(max(1, min(y2, h)))
            if x2 <= x1 or y2 <= y1:
                return None
            return image_bgr[y1:y2, x1:x2].copy()
        except Exception as e:
            print(f"✗ Erro ao extrair estampa (array): {e}")
            return None

# ==================== SIAMESE NETWORK ====================
class SiameseNetworkOptimized(nn.Module):
    """
    Siamese Network com suporte a JIT compilation e half precision.
    """
    def __init__(self, embedding_size=512):
        super(SiameseNetworkOptimized, self).__init__()
        
        # Backbone: ResNet50 pré-treinado
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding layer 
        self.embedding_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # Flag para half precision
        self.use_half = False
    
    def forward_one(self, x):
        """Processa uma imagem e retorna embedding normalizado"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x):
        """Forward para inferência"""
        return self.forward_one(x)
    
    def forward_triplet(self, anchor, positive, negative):
        """Forward para treinamento com triplets"""
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        return anchor_emb, positive_emb, negative_emb
    
    def enable_half_precision(self):
        """Ativa half precision (FP16)"""
        self.half()
        self.use_half = True
    
    def to_torchscript(self, example_input):
        """Converte para TorchScript para inferência mais rápida"""
        self.eval()
        traced = torch.jit.trace(self, example_input)
        return traced

# ==================== TRIPLET LOSS  ====================
class TripletLossBatchAll(nn.Module):
    """
    Triplet Loss com Batch All Mining.
    Implementação vetorizada para máxima performance.
    """
    def __init__(self, margin=0.3):
        super(TripletLossBatchAll, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Calcula Batch All Triplet Loss de forma vetorizada.
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)
        """
        # Calcula matriz de distâncias par-a-par (vetorizado)
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Máscaras para positivos e negativos
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal
        mask_anchor_positive = labels_equal.float()
        mask_anchor_positive.fill_diagonal_(0)
        
        mask_anchor_negative = labels_not_equal.float()
        
        # Calcula loss para todos os triplets válidos
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Aplica máscaras
        mask = mask_anchor_positive.unsqueeze(2) * mask_anchor_negative.unsqueeze(1)
        triplet_loss = triplet_loss * mask
        
        # Remove triplets fáceis
        triplet_loss = F.relu(triplet_loss)
        
        # Conta triplets válidos
        num_positive_triplets = (triplet_loss > 1e-16).float().sum()
        
        # --- INÍCIO DA CORREÇÃO ---
        
        # Pega a soma total da loss. 
        # Esta variável SEMPRE estará conectada ao grafo.
        total_loss_sum = triplet_loss.sum()
        
        if num_positive_triplets > 0:
            # Normaliza pelo número de triplets válidos
            triplet_loss = total_loss_sum / (num_positive_triplets + 1e-16)
        else:
            # Se não há triplets válidos, a loss é 0, 
            # mas usamos total_loss_sum (que é 0.0) para manter o grad_fn
            triplet_loss = total_loss_sum
        
        # --- FIM DA CORREÇÃO ---
            
        return triplet_loss

# ==================== DATASET ====================
class TripletDatasetOptimized(Dataset):
    """
    Dataset com cache e pré-processamento eficiente.
    """
    def __init__(self, root_dir, transform=None, cache_images=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Carrega estrutura
        self.samples = []
        self.class_to_idx = {}
        
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in Config.EXTENSOES_VALIDAS:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        
        print(f"✓ Dataset carregado: {len(self.samples)} imagens, {len(classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Cache de imagens (opcional)
        if self.cache_images and img_path in self.image_cache:
            img = self.image_cache[img_path]
        else:
            img = Image.open(img_path).convert('RGB')
            if self.cache_images:
                self.image_cache[img_path] = img
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ==================== TRAINER COM K-FOLD CROSS-VALIDATION ====================
class TripletTrainerKFold:
    """
    Trainer com K-Fold Cross-Validation e Early Stopping.
    """
    def __init__(self, model, config: Config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        self.criterion = TripletLossBatchAll(margin=config.TRIPLET_MARGIN)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        Path(config.DIR_CHECKPOINTS).mkdir(exist_ok=True)
        
        # Histórico
        self.history = {
            'fold_results': [],
            'best_fold': None,
            'best_val_loss': float('inf')
        }
    
    def train_epoch(self, dataloader):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            embeddings = self.model(images)
            loss = self.criterion(embeddings, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train_with_kfold(self, dataset):
        """
        Treina com K-Fold Cross-Validation.
        """
        print(f"\n{'='*70}")
        print(f"🎯 TREINAMENTO COM {self.config.K_FOLDS}-FOLD CROSS-VALIDATION")
        print(f"{'='*70}\n")
        print(f"📊 Dataset: {len(dataset)} amostras")
        print(f"🔧 Batch Size: {self.config.BATCH_SIZE}")
        print(f"📈 Learning Rate: {self.config.LEARNING_RATE}")
        print(f"⚡ Margin: {self.config.TRIPLET_MARGIN}")
        print(f"💾 Device: {self.device}\n")
        
        kfold = KFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=42)
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\n{'─'*70}")
            print(f"📁 FOLD {fold + 1}/{self.config.K_FOLDS}")
            print(f"{'─'*70}")
            print(f"   Treino: {len(train_ids)} amostras")
            print(f"   Validação: {len(val_ids)} amostras\n")
            
            # Cria DataLoaders
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Reinicializa modelo para cada fold
            self.model = SiameseNetworkOptimized(
                embedding_size=self.config.EMBEDDING_SIZE
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.LEARNING_RATE
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            fold_history = {'train_loss': [], 'val_loss': []}
            
            # Treina fold
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_start = time.time()
                
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                
                epoch_time = time.time() - epoch_start
                
                print(f"   Época [{epoch + 1}/{self.config.NUM_EPOCHS}] - "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Tempo: {epoch_time:.2f}s")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Salva melhor modelo do fold
                    self.save_checkpoint(
                        fold + 1, 
                        epoch + 1, 
                        val_loss, 
                        f"fold_{fold + 1}_best.pth"
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n   ⚠️  Early stopping acionado após {epoch + 1} épocas")
                    break
            
            # Salva resultados do fold
            fold_result = {
                'fold': fold + 1,
                'best_val_loss': best_val_loss,
                'history': fold_history
            }
            self.history['fold_results'].append(fold_result)
            
            print(f"\n   ✓ Fold {fold + 1} concluído - Melhor Val Loss: {best_val_loss:.4f}")
            
            # Atualiza melhor fold global
            if best_val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = best_val_loss
                self.history['best_fold'] = fold + 1
        
        # Resumo final
        print(f"\n{'='*70}")
        print(f"✓ K-FOLD CROSS-VALIDATION CONCLUÍDO")
        print(f"{'='*70}\n")
        
        avg_val_loss = np.mean([f['best_val_loss'] for f in self.history['fold_results']])
        std_val_loss = np.std([f['best_val_loss'] for f in self.history['fold_results']])
        
        print(f"📊 Resultados:")
        print(f"   Melhor Fold: {self.history['best_fold']}")
        print(f"   Melhor Val Loss: {self.history['best_val_loss']:.4f}")
        print(f"   Média Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}\n")
        
        # Salva histórico
        history_path = Path(self.config.DIR_CHECKPOINTS) / "kfold_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, fold, epoch, loss, filename):
        """Salva checkpoint"""
        checkpoint = {
            'fold': fold,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        path = Path(self.config.DIR_CHECKPOINTS) / filename
        torch.save(checkpoint, path)

# ==================== COMPARADOR ====================
class ComparadorEstampasOptimized:
    """
    Sistema de comparação com cache, JIT e batch processing.
    """
    def __init__(self, config: Config, modelo_path: Optional[str] = None):
        self.config = config
        self.detector = DetectorEstampaYOLO(config)
        
        # Carrega modelo
        self.model = SiameseNetworkOptimized(embedding_size=config.EMBEDDING_SIZE)
        
        if modelo_path and Path(modelo_path).exists():
            checkpoint = torch.load(modelo_path, map_location=config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Modelo carregado: {modelo_path}")
        
        self.model.to(config.DEVICE)
        self.model.eval()
        
        # Otimizações
        if config.USE_HALF_PRECISION and config.DEVICE.type == 'cuda':
            self.model.enable_half_precision()
            print("✓ Half Precision (FP16) ativado")
        
        # JIT Compilation
        if config.USE_JIT_COMPILE:
            example_input = torch.randn(1, 3, 224, 224).to(config.DEVICE)
            if config.USE_HALF_PRECISION and config.DEVICE.type == 'cuda':
                example_input = example_input.half()
            
            self.model = self.model.to_torchscript(example_input)
            print("✓ TorchScript JIT compilation ativado")
        
        # Transformações
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Cache de embeddings de referência
        self.embeddings_cache = {}
        if config.CACHE_EMBEDDINGS:
            self._carregar_cache()
        
        print(f"✓ Sistema inicializado no dispositivo: {config.DEVICE}")
    
    def _detector_fingerprint(self) -> str:
        """
        Gera um fingerprint do estado do detector que impacta o embedding:
        - Nome do modelo YOLO (pesos)
        - Parâmetros de inferência (conf, iou, imgsz)
        - Estratégia de seleção (confidence/area)
        - Flags de precisão (FP16)
        """
        try:
            yolo_model_name = str(self.config.YOLO_MODEL)
        except Exception:
            yolo_model_name = "unknown"

        parts = [
            f"yolo={yolo_model_name}",
            f"conf={getattr(self.config, 'YOLO_CONFIDENCE', None)}",
            f"iou={getattr(self.config, 'YOLO_IOU_THRESHOLD', None)}",
            f"imgsz={getattr(self.config, 'YOLO_IMGSZ', None)}",
            f"sel={getattr(self.config, 'YOLO_SELECT_STRATEGY', 'confidence')}",
            f"fp16={bool(getattr(self.config, 'USE_HALF_PRECISION', False))}"
        ]
        return "|".join(parts)

    def _carregar_cache(self):
        """
        Carrega o cache e invalida automaticamente se o fingerprint do detector mudou.
        Formato persistido:
        {
            'meta': {'fingerprint': str, 'version': int},
            'items': { 'path_da_imagem': {'embedding': tensor, 'bbox': tuple} }
        }
        """
        cache_path = Path(self.config.CACHE_FILE)
        self.embeddings_cache = {}
        self._cache_meta = {'fingerprint': self._detector_fingerprint(), 'version': 2}

        if not cache_path.exists():
            return

        try:
            data = torch.load(cache_path, map_location=self.config.DEVICE, weights_only=False)
            if isinstance(data, dict) and 'meta' in data and 'items' in data:
                old_fp = data['meta'].get('fingerprint', '')
                if old_fp == self._cache_meta['fingerprint']:
                    self.embeddings_cache = data['items']
                    self._cache_meta = data['meta']
                    print(f"✓ Cache carregado (compatível): {len(self.embeddings_cache)} embeddings")
                else:
                    print("⚠ Cache inválido (mudança no detector/parâmetros). Reconstrução necessária.")
            else:
                print("⚠ Formato de cache antigo. Será refeito.")
        except Exception as e:
            print(f"⚠ Falha ao carregar cache, será refeito: {e}")

    def _salvar_cache(self):
        """
        Salva cache com metadados para garantir consistência entre runs.
        """
        data = {
            'meta': self._cache_meta,
            'items': self.embeddings_cache
        }
        torch.save(data, self.config.CACHE_FILE)
        """Salva cache de embeddings"""
        torch.save(self.embeddings_cache, self.config.CACHE_FILE)
    
    def processar_imagem(self, imagem_path: str, usar_cache: bool = True) -> Tuple[Optional[torch.Tensor], Optional[Tuple]]:
        """
        Pipeline: Detecta → Extrai → Gera embedding.
        Otimizado para evitar I/O duplicado e overhead de PIL/torchvision.
        COM FALLBACK: Se a detecção falhar, usa a imagem inteira.
        """
        # Cache
        if usar_cache and imagem_path in self.embeddings_cache:
            return self.embeddings_cache[imagem_path]['embedding'], self.embeddings_cache[imagem_path]['bbox']

        # 1) Carrega uma única vez (BGR)
        img_bgr = cv2.imread(imagem_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"   ✗ Falha ao ler {imagem_path}")
            return None, None

        # 2) Detecta com array em memória
        bbox = self.detector.detectar_estampa_array(img_bgr)

        # 3) Extrai recorte via NumPy. Fallback caso necessário
        if bbox is None:
            print(f"   ⚠️  Detecção YOLOv8 falhou para {Path(imagem_path).name}. Usando imagem inteira (fallback)...")
            h, w = img_bgr.shape[:2]
            bbox = (0, 0, w, h)
            crop_bgr = img_bgr
        else:
            crop_bgr = self.detector.extrair_estampa_array(img_bgr, bbox)

        if crop_bgr is None:
            print(f"   ✗ Erro ao extrair estampa (bbox: {bbox})")
            return None, None

        # --- INÍCIO DA VERIFICAÇÃO VISUAL ---
        # Salva o recorte EXATO que será usado
        os.makedirs("./debug_crops", exist_ok=True)
        debug_path = f"./debug_crops/CROP_{Path(imagem_path).name}"
        cv2.imwrite(debug_path, crop_bgr)
        # --- FIM DA VERIFICAÇÃO VISUAL ---

        # 4) Pré-processamento rápido no CPU → GPU
        #    - Resize fixo 224x224 
        crop_bgr = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        #    - Converte para tensor float32 ou float16 diretamente
        tensor = torch.from_numpy(crop_rgb).to(self.config.DEVICE)
        tensor = tensor.permute(2, 0, 1).contiguous()  # HWC->CHW
        tensor = tensor.float()
        if self.config.USE_HALF_PRECISION and self.config.DEVICE.type == 'cuda':
            tensor = tensor.half()

        #    - Normaliza [0,1] e aplica mean/std do ImageNet
        tensor = tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.config.DEVICE, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.config.DEVICE, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) * (1.0 / std)

        tensor = tensor.unsqueeze(0)  # [1, 3, 224, 224]

        # 5) Embedding (inference_mode para máxima velocidade)
        with torch.inference_mode():
            embedding = self.model(tensor)

        # 6) Cache
        if usar_cache:
            self.embeddings_cache[imagem_path] = {
                'embedding': embedding,
                'bbox': bbox
            }
            self._salvar_cache()

        return embedding, bbox
    
    def calcular_similaridade(self, embedding1, embedding2):
        """Calcula similaridade (vetorizado)"""
        # Distância Euclidiana
        # Vetorizado para máxima performance
        distance = F.pairwise_distance(embedding1, embedding2, p=2).item() 
        similarity = 1 - (distance / 2) 
        return similarity, distance # Retorna também distância para info adicional
    
    def comparar_com_referencia(self, imagem_validacao_path: str):
        """
        Compara com referências usando cache e otimizações.
        """
        dir_ref = Path(self.config.DIR_REFERENCIA)
        
        if not dir_ref.exists():
            print(f"✗ Diretório de referência não encontrado")
            return
        
        imagens_ref = [
            str(f) for f in dir_ref.iterdir() 
            if f.suffix.lower() in self.config.EXTENSOES_VALIDAS
        ]
        
        if not imagens_ref:
            print(f"✗ Nenhuma imagem de referência")
            return
        
        print(f"\n{'='*70}")
        print(f"🔍 VALIDAÇÃO")
        print(f"{'='*70}")
        print(f"📁 Imagem: {Path(imagem_validacao_path).name}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}\n")
        
        # Processa imagem de validação
        inicio_total = time.time()
        emb_validacao, bbox_validacao = self.processar_imagem(imagem_validacao_path, usar_cache=False)
        
        if emb_validacao is None:
            print("✗ Falha ao processar imagem\n")
            return
        
        tempo_deteccao = time.time() - inicio_total
        print(f"✓ Detecção: {tempo_deteccao * 1000:.2f}ms")
        
        # Processa referências
        inicio_comparacao = time.time()

        imagens_ref = [
            str(f) for f in dir_ref.iterdir()
            if f.suffix.lower() in self.config.EXTENSOES_VALIDAS
        ]
        if not imagens_ref:
            print("✗ Nenhuma imagem de referência")
            return

        # Lista das que faltam no cache
        faltantes = [p for p in imagens_ref if p not in self.embeddings_cache]

        # Se houver faltantes, processa em batch
        if faltantes:
            bboxes_refs = self.detector.detectar_batch(faltantes)
            for img_ref, bbox_ref in zip(faltantes, bboxes_refs):
                try:
                    if bbox_ref is None:
                        estampa_ref = Image.open(img_ref).convert('RGB')
                        bbox_eff = (0, 0, estampa_ref.width, estampa_ref.height)
                    else:
                        estampa_ref = self.detector.extrair_estampa(img_ref, bbox_ref)
                        bbox_eff = bbox_ref

                    if estampa_ref is None:
                        continue

                    estampa_ref_tensor = self.transform(estampa_ref).unsqueeze(0).to(self.config.DEVICE)
                    if self.config.USE_HALF_PRECISION and self.config.DEVICE.type == 'cuda':
                        estampa_ref_tensor = estampa_ref_tensor.half()

                    with torch.inference_mode():
                        emb_ref = self.model(estampa_ref_tensor)

                    if self.config.CACHE_EMBEDDINGS:
                        self.embeddings_cache[img_ref] = {
                            'embedding': emb_ref,
                            'bbox': bbox_eff
                        }
                except Exception as e:
                    print(f"✗ Falha ao processar referência {img_ref}: {e}")

        # Seleciona melhor match
        melhor_match = None
        melhor_similaridade = -1.0

        for img_ref in imagens_ref:
            item = self.embeddings_cache.get(img_ref)
            if not item:
                continue
            emb_ref = item['embedding']
            similaridade, distancia = self.calcular_similaridade(emb_validacao, emb_ref)
            if similaridade > melhor_similaridade:
                melhor_similaridade = similaridade
                melhor_match = Path(img_ref).name

        if self.config.CACHE_EMBEDDINGS:
            self._salvar_cache()

        tempo_comparacao = time.time() - inicio_comparacao
        
        tempo_comparacao = time.time() - inicio_comparacao
        tempo_total = time.time() - inicio_total
        
        identicas = melhor_similaridade >= self.config.THRESHOLD_SIMILARIDADE
        status = "✓ IDÊNTICAS" if identicas else "✗ DIFERENTES"
        
        print(f"✓ Comparação: {tempo_comparacao * 1000:.2f}ms")
        print(f"\n{'─'*70}")
        print(f"🏆 RESULTADO:")
        print(f"{'─'*70}")
        print(f"   Match: {melhor_match}")
        print(f"   Similaridade: {melhor_similaridade * 100:.2f}%")
        print(f"   Status: {status}")
        print(f"   ⚡ Tempo Total: {tempo_total * 1000:.2f}ms")
        print(f"{'='*70}\n")

    def rebuild_cache_referencias(self) -> None:
        """
        Reconstrói o cache de embeddings das imagens de referência usando SEMPRE o recorte da estampa.
        Usa detecção em batch para maximizar throughput.
        """
        dir_ref = Path(self.config.DIR_REFERENCIA)
        if not dir_ref.exists():
            print("✗ Diretório de referência não encontrado para rebuild_cache_referencias")
            return

        imagens_ref = [
            str(f) for f in dir_ref.iterdir()
            if f.suffix.lower() in self.config.EXTENSOES_VALIDAS
        ]
        if not imagens_ref:
            print("✗ Nenhuma imagem de referência para reconstruir cache")
            return

        print(f"🧱 Reconstruindo cache de referências ({len(imagens_ref)} imagens)...")

        # 1) Detecção em lote
        bboxes_refs = self.detector.detectar_batch(imagens_ref)

        # 2) Extração + embedding (com FP16 se disponível)
        for img_ref, bbox_ref in zip(imagens_ref, bboxes_refs):
            try:
                if bbox_ref is None:
                    # Fallback: usa imagem inteira mas MARCA isso via bbox para diferenciar
                    estampa_ref = Image.open(img_ref).convert('RGB')
                    bbox_eff = (0, 0, estampa_ref.width, estampa_ref.height)
                else:
                    estampa_ref = self.detector.extrair_estampa(img_ref, bbox_ref)
                    bbox_eff = bbox_ref

                if estampa_ref is None:
                    continue

                estampa_ref_tensor = self.transform(estampa_ref).unsqueeze(0).to(self.config.DEVICE)
                if self.config.USE_HALF_PRECISION and self.config.DEVICE.type == 'cuda':
                    estampa_ref_tensor = estampa_ref_tensor.half()

                with torch.inference_mode():
                    emb_ref = self.model(estampa_ref_tensor)

                self.embeddings_cache[img_ref] = {
                    'embedding': emb_ref,
                    'bbox': bbox_eff
                }
            except Exception as e:
                print(f"✗ Falha ao processar referência {img_ref}: {e}")

        self._cache_meta = {'fingerprint': self._detector_fingerprint(), 'version': 2}
        self._salvar_cache()
        print(f"✓ Cache reconstruído: {len(self.embeddings_cache)} embeddings")

# ==================== MONITORAMENTO ====================
class MonitoradorDiretorio(FileSystemEventHandler):
    """Monitora diretório de validação"""
    def __init__(self, comparador: ComparadorEstampasOptimized, config: Config):
        self.comparador = comparador
        self.config = config
        self.processados = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        caminho = Path(event.src_path)
        
        if caminho.suffix.lower() not in self.config.EXTENSOES_VALIDAS:
            return
        
        if str(caminho) in self.processados:
            return
        
        self.processados.add(str(caminho))
        time.sleep(0.5)
        
        self.comparador.comparar_com_referencia(str(caminho))

# ==================== FUNÇÕES PRINCIPAIS ====================
def treinar_modelo_kfold():
    """Treina com K-Fold Cross-Validation"""
    config = Config()
    
    if not Path(config.DIR_TREINAMENTO).exists():
        print(f"✗ Diretório de treinamento não encontrado: {config.DIR_TREINAMENTO}")
        return
    
    # Transformações com data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Carrega dataset
    dataset = TripletDatasetOptimized(config.DIR_TREINAMENTO, transform=transform)
    
    if len(dataset) == 0:
        print("✗ Dataset vazio!")
        return
    
    # Cria modelo
    model = SiameseNetworkOptimized(embedding_size=config.EMBEDDING_SIZE)
    
    # Treina com K-Fold
    trainer = TripletTrainerKFold(model, config)
    history = trainer.train_with_kfold(dataset)
    
    print(f"✓ Treinamento concluído!")

def executar_sistema(usar_modelo_treinado=True):
    """Executa sistema"""
    print("\n" + "="*70)
    print("🚀 SISTEMA DE COMPARAÇÃO DE ESTAMPAS")
    print("="*70 + "\n")
    
    config = Config()
    
    # Cria diretórios
    Path(config.DIR_REFERENCIA).mkdir(exist_ok=True)
    Path(config.DIR_VALIDACAO).mkdir(exist_ok=True)
    Path(config.DIR_YOLO_WEIGHTS).mkdir(exist_ok=True)
    
    # Carrega modelo
    modelo_path = None
    if usar_modelo_treinado:
        # Usa o melhor fold
        history_path = Path(config.DIR_CHECKPOINTS) / "kfold_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            best_fold = history['best_fold']
            modelo_path = Path(config.DIR_CHECKPOINTS) / f"fold_{best_fold}_best.pth"
        
        if not modelo_path or not modelo_path.exists():
            print("⚠️  Modelo treinado não encontrado. Usando modelo base.")
            modelo_path = None
    
    # Inicializa comparador
    comparador = ComparadorEstampasOptimized(config, str(modelo_path) if modelo_path else None)
    
    print(f"\n📁 Diretório de Referência: {config.DIR_REFERENCIA}")
    print(f"📁 Diretório de Validação: {config.DIR_VALIDACAO}")
    print(f"🎯 Threshold: {config.THRESHOLD_SIMILARIDADE * 100:.1f}%\n")
    
    # Pré-processa referências (cache)
    print("🔄 Pré-processando imagens de referência...")
    dir_ref = Path(config.DIR_REFERENCIA)
    if dir_ref.exists():
        for img_ref in dir_ref.iterdir():
            if img_ref.suffix.lower() in config.EXTENSOES_VALIDAS:
                comparador.processar_imagem(str(img_ref), usar_cache=True)
    print("✓ Cache de referências criado\n")
    
    # Configura monitoramento
    event_handler = MonitoradorDiretorio(comparador, config)
    observer = Observer()
    observer.schedule(event_handler, config.DIR_VALIDACAO, recursive=False)
    observer.start()
    
    print(f"👁️  Monitorando: {config.DIR_VALIDACAO}")
    print("⚡ Sistema ativo!\n")
    print("   (Pressione Ctrl+C para encerrar)\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Encerrando...")
        observer.stop()
    
    observer.join()
    print("✓ Sistema encerrado.\n")

def main():
    """Menu principal"""
    import sys
    
    if len(sys.argv) > 1:
        comando = sys.argv[1]
        
        if comando == "treinar":
            treinar_modelo_kfold()
        elif comando == "executar":
            usar_treinado = "--base" not in sys.argv
            executar_sistema(usar_modelo_treinado=usar_treinado)
        else:
            print("Comandos:")
            print("  python sistema.py treinar          # K-Fold Cross-Validation")
            print("  python sistema.py executar         # Rodar o Sistema")
            print("  python sistema.py executar --base  # Modelo base")
    else:
        print("\n" + "="*70)
        print("SISTEMA DE COMPARAÇÃO DE ESTAMPAS")
        print("="*70 + "\n")
        print("1. Treinar modelo (K-Fold Cross-Validation)")
        print("2. Executar sistema (modelo treinado)")
        print("3. Executar sistema (modelo base)")
        print("0. Sair\n")
        
        opcao = input("Opção: ").strip()
        
        if opcao == "1":
            treinar_modelo_kfold()
        elif opcao == "2":
            executar_sistema(usar_modelo_treinado=True)
        elif opcao == "3":
            executar_sistema(usar_modelo_treinado=False)

if __name__ == "__main__":
    main()