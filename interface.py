import customtkinter as ctk
from PIL import Image
import os
import glob
import subprocess
import threading

# Configuração visual do tema
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class InterfaceInspecao(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Inspeção de Estampas")
        self.geometry("1000x750")

        # Pastas de referência
        self.pasta_bandeira = "bandeira"
        self.pasta_imagens = "entrada_monitorada"
        os.makedirs(self.pasta_bandeira, exist_ok=True)
        os.makedirs(self.pasta_imagens, exist_ok=True)

        self.ultima_img_atual = None
        self.ultima_img_bandeira = None
        self.status_icon = None

        self.setup_layout()
        self.atualizar_imagens()
        
        self.log("Sistema iniciado com sucesso. Pronto para produção.")

    def setup_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1) 
        self.grid_columnconfigure(0, weight=1)

        # --- FRAME DAS IMAGENS (TOPO) ---
        self.frame_imagens = ctk.CTkFrame(self)
        self.frame_imagens.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.frame_imagens.grid_columnconfigure(0, weight=1)
        self.frame_imagens.grid_columnconfigure(1, weight=1)

        # Imagem Bandeira
        self.label_titulo_bandeira = ctk.CTkLabel(self.frame_imagens, text="Bandeira (Referência)", font=("Arial", 16, "bold"))
        self.label_titulo_bandeira.grid(row=0, column=0, pady=(10, 0))
        self.label_img_bandeira = ctk.CTkLabel(self.frame_imagens, text="A aguardar Setup...")
        self.label_img_bandeira.grid(row=1, column=0, padx=10, pady=10)

        # Status Visual (Ícone grande)
        self.label_status_visual = ctk.CTkLabel(
            self.frame_imagens, 
            text="", 
            font=("Arial", 80), # Ícone grande
            text_color="gray"
        )
        self.label_status_visual.grid(row=1, column=0, columnspan=2, pady=10)

        # Imagem Atual
        self.label_titulo_atual = ctk.CTkLabel(self.frame_imagens, text="Imagem Atual / Processamento", font=("Arial", 16, "bold"))
        self.label_titulo_atual.grid(row=0, column=1, pady=(10, 0))
        self.label_img_atual = ctk.CTkLabel(self.frame_imagens, text="A aguardar Câmara...")
        self.label_img_atual.grid(row=1, column=1, padx=10, pady=10)

        # --- FRAME DOS BOTÕES (MEIO) ---
        self.frame_botoes = ctk.CTkFrame(self)
        self.frame_botoes.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Dividir o espaço por 4 botões
        for i in range(4):
            self.frame_botoes.grid_columnconfigure(i, weight=1)

        self.btn_setup = ctk.CTkButton(self.frame_botoes, text="📷 Setup (Bandeira)", command=self.rodar_setup)
        self.btn_setup.grid(row=0, column=0, padx=10, pady=15)

        self.btn_contagem1 = ctk.CTkButton(self.frame_botoes, text="🔢 Contagem 1x", fg_color="#d9480f", hover_color="#b83c0c", command=self.rodar_contagem_1x)
        self.btn_contagem1.grid(row=0, column=2, padx=10, pady=15)

        self.btn_contagem2 = ctk.CTkButton(self.frame_botoes, text="🔢 Contagem 2x", fg_color="#d9480f", hover_color="#b83c0c", command=self.rodar_contagem_2x)
        self.btn_contagem2.grid(row=0, column=3, padx=10, pady=15)

        self.btn_diogo = ctk.CTkButton(self.frame_botoes, text="🔍 Análise YOLO", fg_color="#2b8a3e", hover_color="#237032", command=self.rodar_diogo)
        self.btn_diogo.grid(row=0, column=1, padx=10, pady=15)

        # --- FRAME DE TEXTO / LOG (BASE) ---
        self.frame_log = ctk.CTkFrame(self)
        self.frame_log.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew") 
        self.frame_log.grid_columnconfigure(0, weight=1)
        self.frame_log.grid_rowconfigure(1, weight=1) 

        self.label_titulo_log = ctk.CTkLabel(self.frame_log, text="Saída de Dados / Terminal:", font=("Arial", 14, "bold"))
        self.label_titulo_log.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))

        self.caixa_texto = ctk.CTkTextbox(self.frame_log, height=240, font=("Consolas", 14))
        self.caixa_texto.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew") 
        self.caixa_texto.configure(state="disabled")

    # --- FUNÇÃO PARA ESCREVER NA CAIXA DE TEXTO ---
    def log(self, mensagem):
        self.caixa_texto.configure(state="normal")
        self.caixa_texto.insert("end", mensagem + "\n")
        self.caixa_texto.see("end")
        self.caixa_texto.configure(state="disabled")

    # --- FUNÇÕES DE LÓGICA DE IMAGEM ---
    def obter_ultima_imagem(self, pasta):
        """Busca o arquivo mais recente ignorando arquivos ocultos/temporários."""
        try:
            # Lista arquivos com as extensões permitidas
            extensoes = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            arquivos = []
            for ext in extensoes:
                arquivos.extend(glob.glob(os.path.join(pasta, ext)))
            
            if not arquivos:
                return None
            
            # Retorna o arquivo com a data de modificação mais recente
            return max(arquivos, key=os.path.getmtime)
        except Exception:
            return None

    def atualizar_imagens(self):
        """Atualiza os componentes visuais de imagem com tratamento de erro."""
        # 1. Atualizar Imagem Bandeira (Referência)
        arq_bandeira = self.obter_ultima_imagem(self.pasta_bandeira)
        if arq_bandeira and arq_bandeira != self.ultima_img_bandeira:
            try:
                # O uso do 'with' garante que o arquivo seja fechado imediatamente após leitura
                with Image.open(arq_bandeira) as img_pil:
                    # Fazemos um copy() para evitar que o arquivo fique preso em memória
                    img_copy = img_pil.copy()
                    img_ctk = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=(400, 400))
                    self.label_img_bandeira.configure(image=img_ctk, text="")
                    self.ultima_img_bandeira = arq_bandeira
            except Exception as e:
                # Se falhar (arquivo sendo escrito), tentamos novamente no próximo loop
                pass

        # 2. Atualizar Imagem Atual (Monitorada)
        arq_atual = self.obter_ultima_imagem(self.pasta_imagens)
        if arq_atual and arq_atual != self.ultima_img_atual:
            try:
                with Image.open(arq_atual) as img_pil:
                    img_copy = img_pil.copy()
                    img_ctk = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=(400, 400))
                    self.label_img_atual.configure(image=img_ctk, text="")
                    self.ultima_img_atual = arq_atual
            except Exception as e:
                pass

        # Reduzimos o tempo para 300ms para parecer mais "instantâneo"
        self.after(300, self.atualizar_imagens)

    # --- FUNÇÕES DOS BOTÕES (CHAMANDO OS SCRIPTS) ---
    def executar_script_background(self, comando, nome_script):
        def tarefa():
            self.log(f"[{nome_script}] Iniciando...")
            try:
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                processo = subprocess.Popen(
                    comando, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=False, env=env
                )
                
                while True:
                    byte_line = processo.stdout.readline()
                    if not byte_line: break
                    
                    # Decodificação robusta
                    linha = ""
                    for enc in ['utf-8', 'cp1252']:
                        try:
                            linha = byte_line.decode(enc).rstrip()
                            break
                        except: continue

                    if linha:
                        self.after(0, lambda l=linha: self.log(f"[{nome_script}] {l}"))
                        
                        # --- LÓGICA DE STATUS VISUAL ---
                        if "RESULTADO:" in linha:
                            if "IDÊNTICAS" in linha:
                                self.after(0, lambda: self.atualizar_status_visual("✓", "#2b8a3e"))
                            elif "DIFERENTES" in linha:
                                self.after(0, lambda: self.atualizar_status_visual("✗", "#e63946"))
                
                processo.wait()
            except Exception as e:
                self.after(0, lambda: self.log(f"ERR: {str(e)}"))

        threading.Thread(target=tarefa, daemon=True).start()

    def atualizar_status_visual(self, simbolo, cor):
        self.label_status_visual.configure(text=simbolo, text_color=cor)
        # Opcional: Limpar o ícone após 3 segundos
        self.after(3000, lambda: self.label_status_visual.configure(text=""))

    def rodar_setup(self):
        self.log(">> A iniciar Setup (Capturar Bandeira)...")
        self.executar_script_background("python script_setup.py", "SETUP")

    def rodar_contagem_1x(self):
        self.log(">> A iniciar modelo de contagem 1x ...")
        self.executar_script_background("python contagem.py", "COUNT_1X")

    def rodar_contagem_2x(self):
        self.log(">> A iniciar modelo de contagem 2x (Carrossel) ...")
        self.executar_script_background("python contagem_2x.py", "COUNT_2X")

    def rodar_diogo(self):
        self.log(">> A iniciar deteção YOLO e Siamese Network...")
        self.executar_script_background("python C:\\myProjects\\comparador_de_estampas\\sistema_estampas.py executar", "SIAMESE")

if __name__ == "__main__":
    app = InterfaceInspecao()
    app.mainloop()
