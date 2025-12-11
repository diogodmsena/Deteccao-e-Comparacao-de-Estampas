# MANUAL DE INSTALAÇÃO E EXECUÇÃO - SISTEMA DE ESTAMPAS (PORTABLE)

Este pacote contém o executável do "Sistema de Comparação de Estampas".
Ele não requer a instalação manual do Python, mas exige a organização
correta das pastas e dos modelos de Inteligência Artificial.

1. **PRÉ-REQUISITOS DO SISTEMA (COMPUTADOR DE DESTINO)**

   * Placa de Vídeo (Recomendado):
      Para alta performance, recomenda-se uma GPU NVIDIA.
      * Instale os drivers mais recentes da NVIDIA.
      * Instale o CUDA Toolkit (versão 11.8 ou 12.1 recomendada) se o sistema acusar falta de DLLs de aceleração.

   * CPU (Opcional):
      O sistema funcionará apenas com CPU, mas a comparação será mais lenta.

2. **ORGANIZAÇÃO DAS PASTAS E MODELOS (ONDE COLOCAR OS ARQUIVOS)**

   O executável "ComparadorEstampas.exe" busca os modelos e imagens em 
   pastas vizinhas a ele. Você deve garantir a seguinte estrutura:

   Pasta_Do_Sistema/
   │
   ├── ComparadorEstampas.exe      (O executável principal)
   ├── _internal/                  (Arquivos internos do Python/PyInstaller - NÃO MEXA)
   ├── config.json                 (Arquivo de configuração - Veja item 3)
   │
   ├── models/                     <-- CRIE ESTA PASTA
   │   │   (Coloque aqui os arquivos de peso da IA)
   │   ├── estampa_yolov8n_best.pt    (Modelo YOLO treinado)
   │   └── fold_1_best.pth            (Modelo Siamesa treinado)
   │
   ├── referencias/                <-- CRIE ESTA PASTA
   │   │   (Coloque aqui as imagens .jpg/.png que servem de gabarito)
   │   ├── estampa_original_A.jpg
   │   └── estampa_original_B.jpg
   │
   └── entrada/                    <-- CRIE ESTA PASTA
         (Pasta vazia inicial. O sistema monitorará esta pasta)

3. **ARQUIVO DE CONFIGURAÇÃO (config.json)**

   Crie um arquivo chamado "config.json" na mesma pasta do .exe.
   Se ele não existir, o sistema usará padrões internos. 

   Exemplo de conteúdo recomendado para o config.json:

   ```json
   {
      "DIR_REFERENCIA": "./referencias",
      "DIR_VALIDACAO": "./entrada",
      "DIR_CHECKPOINTS": "./models",
      "DIR_YOLO_WEIGHTS": "./models",
      "YOLO_MODEL": "estampa_yolov8n_best.pt",
      "SIAMESE_MODEL": "fold_1_best.pth",
      "THRESHOLD_SIMILARIDADE": 0.85,
      "DEVICE": "cuda" 
   }
   ```

   * Nota: Mude "DEVICE": "cpu" se o computador não tiver placa NVIDIA.

4. **COMO EXECUTAR**

   Modo 1: Clique Duplo
      * Basta clicar duas vezes em "ComparadorEstampas.exe".
      * Uma janela preta (console) abrirá mostrando o log do sistema.
      * Selecione a opção "2. Executar sistema (modelo treinado)".

   Modo 2: Via Prompt de Comando (CMD/PowerShell) - Recomendado para logs
      1. Abra o terminal na pasta do sistema.
      2. Digite: .\ComparadorEstampas.exe executar

      Isso iniciará o monitoramento da pasta "./entrada".
      Qualquer imagem salva nessa pasta será processada automaticamente.

5. **RESOLUÇÃO DE PROBLEMAS**

   * Erro "ModuleNotFoundError": Geralmente resolvido pelo PyInstaller, mas
   verifique se a pasta "_internal" está completa junto ao .exe.
   * Erro "CUDA not available": Edite o config.json para "DEVICE": "cpu"
   ou instale os drivers NVIDIA.
   * Erro "Model not found": Verifique se o nome do arquivo dentro da pasta "models" é EXATAMENTE igual ao nome no config.json.
