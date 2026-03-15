import cv2
import os
import time

pasta_bandeira = "bandeira"
os.makedirs(pasta_bandeira, exist_ok=True)

print("A iniciar a câmara para capturar a Bandeira...")

# Mude o 0 para o índice da sua câmara industrial, se aplicável
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("ERRO: Não foi possível aceder à câmara para o Setup.")
    exit()

print("A focar a imagem... Aguarde 2 segundos.")
time.sleep(2) # Tempo para a câmara ajustar brilho e foco automático

ret, frame = cap.read()

if ret:
    caminho_salvar = os.path.join(pasta_bandeira, "bandeira_referencia.jpg")
    cv2.imwrite(caminho_salvar, frame)
    print("SUCESSO: Foto da bandeira capturada com sucesso!")
    print(f"Imagem guardada em: {caminho_salvar}")
else:
    print("ERRO: Ocorreu uma falha ao capturar a imagem da câmara.")

cap.release()
