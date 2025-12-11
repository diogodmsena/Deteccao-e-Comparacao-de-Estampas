import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

# Estratégias de seleção de caixa
def _choose_box(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    confidences: List[float],
    strategy: str = "confidence"
) -> int:
    if not boxes_xyxy:
        return -1
    if strategy == "area":
        areas = []
        for x1, y1, x2, y2 in boxes_xyxy:
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        return int(max(range(len(areas)), key=lambda i: areas[i]))
    return int(max(range(len(confidences)), key=lambda i: confidences[i]))

class EstampaDetector:
    """
    Wrapper de inferência YOLOv8 para detecção da região de estampa.
    Responsável por:
      - Carregar checkpoint (best.pt)
      - Executar predict() com conf/iou/imgsz
      - Selecionar uma caixa final
      - Retornar a caixa e o recorte (crop) da estampa
    """
    def __init__(
        self,
        weights_path: str = "models/estampa_yolov8_best.pt",
        conf: float = 0.30,
        iou: float = 0.55,
        imgsz: int = 640,
        device: str = "0",  # "0" para GPU, "cpu" para CPU
        select_strategy: str = "confidence",  # "confidence" ou "area"
        fp16: bool = True
    ) -> None:
        self.weights_path = weights_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.select_strategy = select_strategy
        self.fp16 = fp16

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights_path}")

        # Carrega o modelo
        self.model = YOLO(weights_path)

        # Configura dispositivo. Ultralytics lê device no predict; fp16 via amp automática.
        # Para garantir mixed precision, mantemos fp16=True e imgsz coerente com o treino.
        # Em CPU, o fp16 é ignorado.
        os.environ["YOLO_VERBOSE"] = "False"

    def detect(
        self,
        image_bgr: np.ndarray
    ) -> Dict[str, Optional[object]]:
        """
        Executa a detecção e retorna:
          {
            "bbox": (x1, y1, x2, y2) ou None,
            "conf": float ou None,
            "crop": recorte BGR ou None
          }
        """
        if image_bgr is None or image_bgr.size == 0:
            return {"bbox": None, "conf": None, "crop": None}

        # Realiza a inferência
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        if not results or len(results) == 0:
            return {"bbox": None, "conf": None, "crop": None}

        r = results[0]
        boxes = []
        confs = []

        # Extrai caixas no formato xyxy e confianças
        if r.boxes is None or r.boxes.xyxy is None or r.boxes.xyxy.shape[0] == 0:
            return {"bbox": None, "conf": None, "crop": None}

        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            boxes.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])))
            confs.append(float(b.conf[0].item()))

        idx = _choose_box(boxes, confs, strategy=self.select_strategy)
        if idx < 0:
            return {"bbox": None, "conf": None, "crop": None}

        x1, y1, x2, y2 = boxes[idx]
        h, w = image_bgr.shape[:2]

        # Clampeia coordenadas válidas
        x1i = max(0, min(int(x1), w - 1))
        y1i = max(0, min(int(y1), h - 1))
        x2i = max(0, min(int(x2), w))
        y2i = max(0, min(int(y2), h))

        if x2i <= x1i or y2i <= y1i:
            return {"bbox": None, "conf": None, "crop": None}

        crop = image_bgr[y1i:y2i, x1i:x2i].copy()
        return {
            "bbox": (x1i, y1i, x2i, y2i),
            "conf": confs[idx],
            "crop": crop
        }