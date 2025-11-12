import cv2
import numpy as np

# Importación del video
video_path = 'C:/Users/julio/Documents/UNSTA (laptop local)/Fisica II files/WhatsApp Video 2025-09-29 at 20.17.52.mp4'
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abrió correctamente.
if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

# Inicializa el sustractor de fondo.
# Este método ayuda a separar los objetos en movimiento del fondo estático.
# Se utiliza el algoritmo MOG2 que es efectivo para este propósito.
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Lee un fotograma del video.
    ret, frame = cap.read()

    # Si 'ret' es False, significa que hemos llegado al final del video.
    if not ret:
        break

    # Aplica el sustractor de fondo para obtener la máscara de primer plano.
    # Esta máscara mostrará las áreas donde se detecta movimiento.
    foreground_mask = background_subtractor.apply(frame)

    # Opcional: Aplica operaciones morfológicas para reducir el ruido.
    # Esto ayuda a eliminar pequeñas imperfecciones en la máscara.
    kernel = np.ones((5,5),np.uint8)
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    # Encuentra los contornos de los objetos en movimiento en la máscara.
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja los contornos en el fotograma original.
    for contour in contours:
        # Puedes filtrar los contornos por área para ignorar movimientos muy pequeños.
        if cv2.contourArea(contour) > 500: # El valor 500 es un umbral que puedes ajustar.
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Muestra el fotograma original con los contornos.
    cv2.imshow('Analisis del Motor - Movimiento', frame)

    # Muestra la máscara de primer plano.
    cv2.imshow('Mascara de Movimiento', foreground_mask)

    # Espera por la tecla 'q' para salir del bucle.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libera el objeto de captura de video y cierra todas las ventanas.
cap.release()
cv2.destroyAllWindows()