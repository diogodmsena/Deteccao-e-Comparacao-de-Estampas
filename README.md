# Sistema de Detecção e Comparação de Estampas

Este documento README.md fornece uma visão abrangente do Sistema de Detecção e Comparação de Estampas, um projeto de visão computacional avançado focado na análise de padrões em camisetas e itens de vestuário.

## Descrição do Projeto

O Sistema de Detecção e Comparação de Estampas é uma solução de visão computacional projetada para identificar e comparar estampas ou designs em camisetas e peças de vestuário. Utilizando uma combinação robusta de YOLOv8 para detecção de objetos e Redes Neurais Siamesas para comparação de padrões, o sistema oferece alta precisão e desempenho otimizado. Ele incorpora diversas otimizações, como compilação TorchScript JIT, inferência em meia precisão (FP16), processamento em lote e cache de embeddings, garantindo uma correspondência de padrões eficiente e escalável.

## Principais Funcionalidades

* **Detecção de Estampas:** Utiliza YOLOv8 (com opção de RT-DETR) para detecção precisa de estampas/designs, com fallback para imagem completa caso nenhuma estampa seja detectada.
* **Geração de Embeddings:** Rede Neural Siamesa com backbone ResNet50 para gerar representações vetoriais (embeddings) de estampas.
* **Otimização de Inferência (TorchScript):** Compilação Just-In-Time (JIT) com TorchScript para aceleração da inferência do modelo.
* **Meia Precisão (FP16):** Suporte a inferência em FP16 (half-precision) para GPUs compatíveis, reduzindo o uso de memória e aumentando a velocidade.
* **Processamento em Lote:** Otimização do throughput através do processamento simultâneo de múltiplas imagens.
* **Cache de Embeddings:** Sistema de cache para armazenar embeddings gerados, evitando recalculos redundantes e acelerando comparações.
* **Validação Cruzada (K-Fold):** Treinamento robusto do modelo Siamese utilizando validação cruzada K-Fold.
* **Early Stopping:** Mecanismo de parada antecipada para prevenir overfitting durante o treinamento.
* **Triplet Loss com Batch All Mining:** Função de perda Triplet Loss implementada com estratégia de Batch All Mining para um treinamento eficiente.
* **Monitoramento de Diretórios:** Capacidade de monitorar diretórios para processamento e comparação automática de novos arquivos (via biblioteca `watchdog`).
* **Interface de Linha de Comando (CLI):** Ferramentas de linha de comando para treinamento do modelo e execução do sistema.

## Componentes de Arquitetura

1. **`DetectorEstampaYOLO`**: Módulo responsável pela integração com YOLOv8, detecção em lote e capacidades de fine-tuning.
2. **`SiameseNetworkOptimized`**: Implementação da Rede Neural Siamesa baseada em ResNet50, com camadas de embedding customizadas, conversão para TorchScript e suporte a FP16.
3. **`ComparadorEstampasOptimized`**: Motor de comparação de estampas, incluindo o sistema de cache, métricas de distância e cálculo de pontuação de similaridade.
4. **`TripletTrainerKFold`**: Classe para treinamento do modelo Siamese utilizando K-Fold Cross-Validation e early stopping.
5. **`TripletDatasetOptimized`**: Classe de dataset customizada otimizada com cache em memória para datasets Triplet.
6. **`TripletLossBatchAll`**: Implementação corrigida da função de perda Triplet Loss com Batch All Mining.
7. **`MonitoradorDiretorio`**: Módulo para monitoramento contínuo de diretórios em busca de novas imagens para processamento.
8. **`Config`**: Classe centralizada para gerenciar todos os parâmetros de configuração do sistema.

### Requisitos Técnicos

* Python 3.8 ou superior
* PyTorch
* torchvision
* PIL (Pillow)
* numpy
* scikit-learn
* watchdog
* ultralytics (para YOLOv8)
* GPU compatível com CUDA (altamente recomendado para desempenho ideal)

-----

## Instalação

Siga os passos abaixo para configurar o ambiente e instalar todas as dependências.

1. **Crie e ative um ambiente virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # ou
    venv\Scripts\activate     # Windows
    ```

2. **Instale PyTorch com suporte a CUDA (Crítico para Desempenho):**

    O desempenho do sistema depende **criticamente** da instalação correta do PyTorch com suporte a CUDA. Se a versão "somente CPU" for instalada por engano, o sistema funcionará, mas será **extremamente lento**, pois todo o processamento será feito pela CPU.

    Visite o [site oficial do PyTorch](https://pytorch.org/get-started/locally/) para obter o comando exato para sua versão de CUDA.

    *Exemplo para CUDA 12.1 (GPUs mais recentes):*

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

    *Exemplo para CUDA 11.8:*

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

    *Se você não tiver uma GPU ou preferir usar apenas CPU (Não recomendado para produção):*

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

3. **Instale as demais dependências:**

    ```bash
    pip install ultralytics watchdog scikit-learn numpy pillow
    ```

4. **(IMPORTANTE) Verifique se a GPU está habilitada:**

    Após instalar tudo, execute o comando abaixo no seu terminal (com o ambiente `venv` ativado) para confirmar que o PyTorch pode "enxergar" sua GPU.

    ```bash
    python -c "import torch; print(f'GPU Habilitada: {torch.cuda.is_available()}')"
    ```

      * Se o resultado for `GPU Habilitada: True`, sua instalação está correta e o sistema usará a GPU para aceleração.
      * Se o resultado for `GPU Habilitada: False`, o PyTorch não está detectando sua GPU. Você provavelmente instalou a versão "somente CPU". Reinstale o PyTorch usando o comando correto com CUDA (do passo 2).

### Estrutura do Dataset

A organização do dataset é crucial para o treinamento e operação do sistema.

#### Dataset para Detecção (YOLOv8)

Para o treinamento do detector YOLOv8, o dataset deve seguir o formato YOLO:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

Onde cada arquivo `.txt` contém as anotações para sua respectiva imagem, no formato `class_id x_center y_center width height` (valores normalizados).

#### Dataset para Treinamento Triplet (Siamese Network)

Para o treinamento da Rede Siamese com Triplet Loss, o dataset pode ser organizado por classes, permitindo que o `TripletDatasetOptimized` gere os triplets (âncora, positivo, negativo) dinamicamente:

```
triplet_dataset/
├── class_A/
│   ├── shirt_A_1.jpg
│   ├── shirt_A_2.jpg
│   └── ...
├── class_B/
│   ├── shirt_B_1.jpg
│   ├── shirt_B_2.jpg
│   └── ...
└── ...
```

Cada subdiretório representa uma classe de estampa distinta.

### Configuração

A classe `Config` centraliza todos os parâmetros configuráveis do sistema. Recomenda-se criar um arquivo `config.py` ou `config.json` e carregá-lo na inicialização.

Principais parâmetros configuráveis:

* **Caminhos de Diretórios:**
* `DATA_DIR`: Diretório raiz dos datasets.
* `CHECKPOINT_DIR`: Diretório para salvar os checkpoints dos modelos.
* `MONITOR_INPUT_DIR`: Diretório monitorado para novas imagens a serem processadas.
* `MONITOR_OUTPUT_DIR`: Diretório para salvar os resultados das comparações.
* **Parâmetros do Modelo:**
* `DETECTION_THRESHOLD`: Limiar de confiança para detecções do YOLOv8.
* `SIMILARITY_THRESHOLD`: Limiar de similaridade para considerar duas estampas como iguais.
* `YOLO_MODEL_PATH`: Caminho para o modelo YOLOv8 pré-treinado ou fine-tuned.
* `SIAMESE_MODEL_PATH`: Caminho para o modelo Siamese pré-treinado.
* **Flags de Otimização:**
* `USE_TORCHSCRIPT`: Booleano para ativar a compilação TorchScript.
* `USE_FP16`: Booleano para ativar inferência em FP16.
* `USE_EMBEDDING_CACHE`: Booleano para ativar o sistema de cache de embeddings.
* **Parâmetros de Treinamento (Siamese Network):**
* `EPOCHS`: Número total de épocas de treinamento.
* `BATCH_SIZE`: Tamanho do lote para treinamento.
* `LEARNING_RATE`: Taxa de aprendizado.
* `K_FOLDS`: Número de folds para validação cruzada K-Fold.
* `MARGIN_TRIPLET_LOSS`: Margem para a função de perda Triplet Loss.

### Instruções de Uso

O sistema oferece uma interface de linha de comando para suas principais operações.

#### Treinando o Modelo Siamese

Para treinar a Rede Neural Siamesa utilizando validação cruzada K-Fold:

```bash
python main.py train_siamese --config_path ./config.py
```

Você pode especificar outros parâmetros de treinamento via linha de comando ou no arquivo de configuração.

#### Executando o Sistema de Comparação

Para executar o sistema de comparação, seja manualmente ou via monitoramento de diretório:

```bash
python sistema_estampas.py executar
```

* `--input_path`: Caminho para uma imagem ou diretório de imagens a serem processadas.
* `--output_path`: Diretório onde os resultados da comparação serão salvos.
* `--monitor True`: Ativa o monitoramento contínuo do `input_path` (se for um diretório).
* `--config_path`: Caminho para o arquivo de configuração.

#### Fine-tuning do YOLOv8

Para fine-tunear o modelo de detecção YOLOv8 em seu dataset customizado:

```bash
python main.py finetune_yolo --data_yaml ./yolo_dataset.yaml --epochs 50 --batch_size 16 --config_path ./config.py
```

* `--data_yaml`: Caminho para o arquivo `.yaml` do dataset YOLO.
* `--epochs`: Número de épocas para o fine-tuning.
* `--batch_size`: Tamanho do lote.

#### Opções da Interface de Linha de Comando

A função `main()` do sistema aceita vários subcomandos e argumentos, como `train_siamese`, `run_comparison`, `finetune_yolo`, entre outros, para controlar as diferentes funcionalidades do sistema. Use `--help` para ver todas as opções:

```bash
python main.py --help
python main.py train_siamese --help
```

### Otimizações de Desempenho

O sistema foi projetado com várias otimizações para garantir alta performance:

* **Compilação JIT com TorchScript:**
* **Benefício:** Permite exportar modelos PyTorch para um formato otimizado que pode ser executado independentemente do código Python, resultando em inferência mais rápida, especialmente em ambientes de produção e em C++.
* **Requisitos:** O modelo deve ser "scriptable" (compatível com as operações suportadas pelo TorchScript).
* **Inferência em Meia Precisão (FP16):**
* **Benefício:** Reduz pela metade o uso de memória da GPU e acelera a computação em GPUs com Tensor Cores, como as GPUs NVIDIA modernas, sem perda significativa de precisão para muitos modelos de visão.
* **Requisitos:** GPU compatível com FP16 (NVIDIA Turing, Ampere, Ada Lovelace, Hopper, etc.) e PyTorch configurado com CUDA.
* **Processamento em Lote (Batch Processing):**
* **Benefício:** Agrupa múltiplas entradas (imagens) para processamento simultâneo pelos modelos. Isso aproveita a paralelização da GPU, resultando em maior throughput (imagens processadas por segundo) em comparação com o processamento de imagens uma a uma.
* **Requisitos:** Necessidade de agrupar e redimensionar imagens para o mesmo formato.
* **Cache de Embeddings:**
* **Benefício:** Para imagens já processadas e cujos embeddings já foram calculados, o sistema armazena esses embeddings. Em comparações subsequentes, se a mesma imagem for encontrada, o embedding é recuperado do cache em vez de ser recalculado, economizando tempo computacional significativo.
* **Requisitos:** Armazenamento persistente ou em memória para o cache, com uma estratégia de invalidação ou limite de tamanho.

### Estrutura de Arquivos

A estrutura de diretórios esperada para o projeto é a seguinte:

```
.
├── config.py                   # Arquivo de configuração
├── main.py                     # Ponto de entrada da CLI
├── src/
│   ├── arch/                   # Definições de arquitetura dos modelos
│   │   ├── siamese_network.py
│   │   └── yolo_detector.py
│   ├── data/                   # Classes de dataset e pré-processamento
│   │   ├── triplet_dataset.py
│   │   └── transforms.py
│   ├── core/                   # Lógica central do sistema
│   │   ├── comparator.py
│   │   ├── monitor.py
│   │   └── trainer.py
│   └── utils/                  # Funções utilitárias e ferramentas
│       ├── cache_manager.py
│       ├── cli_parser.py
│       └── metrics.py
├── data/                       # Diretório para datasets (conforme Estrutura do Dataset)
│   ├── yolo_detection_dataset/
│   └── triplet_training_dataset/
├── checkpoints/                # Modelos treinados e checkpoints
│   ├── yolo_best.pt
│   └── siamese_best.pth
├── monitored_input/            # Diretório para monitoramento automático
├── results/                    # Resultados das comparações
└── README.md
```

### Como Funciona

O sistema segue um fluxo de trabalho modular para detecção e comparação de estampas:

1. **Ingestão de Imagem:** Novas imagens são recebidas, seja através de carregamento manual ou detecção automática pelo monitoramento de diretório.
2. **Detecção de Estampa:** O `DetectorEstampaYOLO` processa a imagem para identificar e extrair as regiões de interesse (ROI) que contêm as estampas. Se nenhuma estampa for detectada, a imagem completa pode ser usada como fallback.
3. **Geração de Embedding:** As estampas detectadas (ou a imagem completa) são passadas para a `SiameseNetworkOptimized`, que gera um vetor de características (embedding) de alta dimensão, representando a essência visual da estampa. O cache de embeddings é consultado e atualizado neste estágio.
4. **Comparação de Similaridade:** O `ComparadorEstampasOptimized` compara os embeddings recém-gerados com os embeddings de estampas já conhecidas (armazenados ou de outras imagens no lote), utilizando métricas de distância (ex: distância euclidiana, cosseno) para calcular pontuações de similaridade.
5. **Saída de Resultados:** Os resultados da comparação, incluindo as pontuações de similaridade e as identidades das estampas correspondentes, são apresentados e/ou salvos em um formato configurado.

### Processo de Treinamento

O treinamento da Rede Neural Siamesa é um componente crítico para a eficácia do sistema:

* **Triplet Loss:** A Rede Siamesa é treinada usando a função de perda Triplet Loss. Esta perda visa garantir que a distância no espaço de embedding entre um par (âncora, positivo) seja menor do que a distância entre um par (âncora, negativo) por uma margem definida. A implementação `TripletLossBatchAll` seleciona os "hard triplets" dentro de cada lote para otimizar o aprendizado.
* **Validação Cruzada K-Fold:** Para garantir a robustez e generalização do modelo, o treinamento é realizado utilizando validação cruzada K-Fold. O dataset é dividido em K subconjuntos, e o treinamento é iterado K vezes, usando K-1 subconjuntos para treinamento e 1 para validação, o que ajuda a obter uma avaliação mais confiável do desempenho do modelo.
* **Mecanismo de Early Stopping:** Durante o treinamento, é implementado um mecanismo de early stopping. Se a perda de validação não melhorar por um número predefinido de épocas (patience), o treinamento é interrompido. Isso previne o overfitting e otimiza o tempo de treinamento.
* **Pré-processamento do Dataset:** As imagens do dataset são pré-processadas (redimensionamento, normalização, aumentações de dados) antes de serem alimentadas na rede para melhorar a resiliência e o desempenho do modelo.

### Tratamento de Erros

O sistema incorpora mecanismos robustos de tratamento de erros e logging. Erros inesperados são capturados e registrados para facilitar a depuração. As operações críticas são envolvidas em blocos `try-except` para garantir a estabilidade e a continuidade do processamento, mesmo diante de falhas pontuais (ex: arquivo corrompido, modelo não encontrado).

### Requisitos de Hardware

* **GPU Recomendada:** Uma GPU NVIDIA com pelo menos 8GB de VRAM (ex: RTX 3060, RTX 4060, A2000 ou superior) é altamente recomendada para aproveitar as otimizações de FP16 e processamento em lote, garantindo a melhor performance e tempos de inferência mais rápidos.
* **Suporte a CPU:** O sistema pode ser executado em modo somente CPU, mas com desempenho significativamente reduzido, especialmente para grandes volumes de dados ou modelos complexos.
* **Requisitos de Memória:** Para datasets e modelos maiores, recomenda-se ter pelo menos 16GB de RAM do sistema.

### Melhorias Futuras Sugeridas

* **Modelos Adicionais:** Explorar outros modelos de detecção (ex: Faster R-CNN, EfficientDet) ou redes Siamese com diferentes backbones (ex: Vision Transformers) para comparação de desempenho.
* **Processamento de Vídeo em Tempo Real:** Adaptar o sistema para processar streams de vídeo em tempo real para detecção e comparação contínua.
* **Web API:** Desenvolver uma API RESTful para integrar o sistema a aplicações web e móveis, permitindo upload de imagens e recuperação de resultados via HTTP.
* **Interface Gráfica (GUI):** Criar uma interface gráfica do usuário para facilitar a interação com o sistema para usuários não técnicos.
* **Otimização de Quantização:** Implementar quantização de modelo (ex: INT8) para reduzir ainda mais o tamanho do modelo e acelerar a inferência em hardware específico.
* **Indexação de Embeddings:** Utilizar bibliotecas de busca de vizinhos mais próximos eficientes (ex: Faiss) para acelerar a comparação de embeddings em larga escala.

### Licença

---

### Contribuidores

---

### Contato

---
