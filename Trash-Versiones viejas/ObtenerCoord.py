import cv2
import sys

# --- CONFIGURACIÓN ---
# ¡Pon aquí la ruta a tu video COMPLETO!
VIDEO_PATH_ORIGINAL = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4'
# ---------------------


print(f"Cargando video desde: {VIDEO_PATH_ORIGINAL}")
cap = cv2.VideoCapture(VIDEO_PATH_ORIGINAL)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    sys.exit()

# Leemos solo el primer frame
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer frame.")
    sys.exit()

print("\n--- INSTRUCCIONES ---")
print("1. Se abrirá una ventana con el video.")
print("2. Haz clic y arrastra con el mouse para dibujar el rectángulo (ROI).")
print("3. Cuando estés listo, presiona la tecla 'ENTER' o 'ESPACIO'.")
print("4. (Puedes presionar 'c' para cancelar y volver a dibujar).")
print("-----------------------")

# Abrir el selector de ROI
# fromCenter=False significa que dibujamos desde la esquina
roi = cv2.selectROI("ASISTENTE: Dibuja la ROI y presiona ENTER", frame, fromCenter=False, showCrosshair=True)

# Cerramos la ventana
cv2.destroyAllWindows()
cap.release()

# --- RESULTADOS ---
print("\n¡ROI seleccionada!")
print(f"Coordenadas (x, y, w, h): {roi}")

# El selector de ROI devuelve (x, y, ancho, alto)
# Necesitamos convertirlo a (x1, y1, x2, y2)
x1 = roi[0]
y1 = roi[1]
x2 = roi[0] + roi[2] # x + ancho
y2 = roi[1] + roi[3] # y + alto

print("\n--- ¡COPIA ESTA LÍNEA EN TU SCRIPT PRINCIPAL! ---")
print(f"ROI_PARA_SCRIPT = ({x1}, {y1}, {x2}, {y2})")
print("-------------------------------------------------")