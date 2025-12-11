import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from src.detector.estampa_detector import EstampaDetector

# Dependendo do seu projeto, substitua por seu encoder real Siamesa
class DummySiameseEncoder:
    def __init__(self) -> None:
        pass

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        # Placeholder: retorne vetor aleatório com dimensão fixa
        rng = np.random.default_rng(123)
        return rng.normal(size=(512,)).astype(np.float32)

class EstampaPipeline:
    def __init__(
        self,
        detector_weights: str = "models/estampa_yolov8_best.pt",
        conf: float = 0.30,
        iou: float = 0.55,
        imgsz: int = 640,
        device: str = "0",
        select_strategy: str = "confidence"
    ) -> None:
        self.detector = EstampaDetector(
            weights_path=detector_weights,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            select_strategy=select_strategy,
            fp16=True
        )
        self.encoder = DummySiameseEncoder()

    def process_image(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"ok": False, "msg": f"Falha ao ler {image_path}"}

        det = self.detector.detect(img)
        crop = det["crop"]
        if crop is None:
            # Opcional: fallback com log
            crop = img
            used_fallback = True
        else:
            used_fallback = False

        embedding = self.encoder.embed(crop)
        return {
            "ok": True,
            "bbox": det["bbox"],
            "conf": det["conf"],
            "used_fallback": used_fallback,
            "embedding": embedding
        }

if __name__ == "__main__":
    # Exemplo de execução
    pipeline = EstampaPipeline(
        detector_weights="models/best.pt",
        conf=0.30,
        iou=0.55,
        imgsz=640,
        device="0",
        select_strategy="confidence"
    )
    result = pipeline.process_image("exemplos/entrada.jpg")
    print(result)