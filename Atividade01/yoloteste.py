import cv2

# Inicializa o detector HOG para detecção de pessoas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Captura o vídeo da webcam (ou pode substituir pelo caminho de um vídeo)
cap = cv2.VideoCapture(0)  # Use 0 para a webcam padrão ou substitua por um caminho de vídeo

while True:
    # Captura frame-by-frame
    ret, frame = cap.read()
    
    # Verifica se o frame foi capturado corretamente
    if not ret:
        print("Falha na captura do frame")
        break

    # Redimensiona o frame para melhorar a velocidade de processamento
    frame = cv2.resize(frame, (640, 480))

    # Detecta pessoas no frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(5, 5), padding=(18, 18), scale=1.05)

    # Desenha as caixas delimitadoras ao redor das pessoas detectadas com a cor verde
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostra o frame processado
    cv2.imshow('Detecção de Pessoas', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()