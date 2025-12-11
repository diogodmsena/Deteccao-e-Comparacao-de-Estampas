import sys
from pathlib import Path

# Ajuste o import para apontar para seu módulo real
# Ex.: from sistema_estampas import Config, DetectorEstampaYOLO
from sistema_estampas import Config, DetectorEstampaYOLO

def main() -> int:
    """
    Smoke test do YOLOv8 isolado:
      - Carrega Config e Detector
      - Executa detecção em uma imagem
      - Extrai e reporta o crop da estampa
    Retornos:
      0 = sucesso
      1 = erro
    """
    if len(sys.argv) < 2:
        print("Uso: python dev_check.py caminho/para/imagem.jpg")
        return 1

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"✗ Arquivo não encontrado: {img_path}")
        return 1

    try:
        config = Config()
        det = DetectorEstampaYOLO(config)
    except Exception as e:
        print(f"✗ Falha ao inicializar detector: {e}")
        return 1

    try:
        bbox = det.detectar_estampa(str(img_path))
        print("BBox:", bbox)
        if bbox is None:
            print("⚠ Nenhuma estampa detectada (fallback seria usar a imagem inteira).")
            return 0

        estampa = det.extrair_estampa(str(img_path), bbox)
        if estampa is None:
            print("✗ Falha ao extrair estampa a partir do bbox.")
            return 1

        print("Crop size:", estampa.size)
        # Visualização opcional:
        # estampa.show()
        return 0

    except Exception as e:
        print(f"✗ Erro durante o teste: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())