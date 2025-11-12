# --- SCRIPT: selector_color.py (CORREGIDO - AHORA SÍ SE DETIENE) ---
import cv2
import numpy as np

# --- CONFIGURACIÓN ---
# !! Asegúrate de que la ruta sea la misma que en tu script principal !!
#Local:
# video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
#Drive:
video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'

window_name = "Selector de Color (Frame por Frame)"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Función para el clic del mouse ---
def get_pixel_color(event, x, y, flags, param):
    """
    Esta función se activa al hacer clic y muestra los valores HSV del píxel.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param['frame']
        bgr_pixel = frame[y, x]
        
        # Convertir el píxel a HSV para obtener los valores
        pixel_image = np.uint8([[bgr_pixel]])
        hsv_pixel = cv2.cvtColor(pixel_image, cv2.COLOR_BGR2HSV)[0][0]
        
        print("--------------------")
        print(f"Coordenadas: (x={x}, y={y})")
        print(f"Color BGR: [B={bgr_pixel[0]}, G={bgr_pixel[1]}, R={bgr_pixel[2]}]")
        print(f"Color HSV: [H={hsv_pixel[0]}, S={hsv_pixel[1]}, V={hsv_pixel[2]}] <-- ¡Anota estos valores!")
        print("--------------------")

# --- Función para la barra de desplazamiento (AHORA CONTROLA TODO) ---
def on_trackbar(val):
    """
    Esta función se ejecuta CADA VEZ que mueves la barra.
    Busca el frame, lo muestra y prepara la función de clic para ese frame.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        # Mostramos el frame estático
        cv2.imshow(window_name, frame)
        # Asociamos la función de clic A ESTE frame específico
        cv2.setMouseCallback(window_name, get_pixel_color, {'frame': frame})

# --- Configuración de la ventana y ejecución ---
cv2.namedWindow(window_name)
cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)

print("\n--- INSTRUCCIONES (VERSIÓN CORREGIDA) ---")
print("1. Usa la barra 'Frame' para encontrar una imagen clara de la estela.")
print("2. El video AHORA se quedará quieto en el frame que elijas.")
print("3. Haz clic en diferentes partes de la estela para ver sus valores HSV.")
print("4. Anota los valores mínimos y máximos de H, S y V.")
print("5. Presiona 'q' para salir.")

# Mostrar el primer frame para empezar
on_trackbar(0)

# Este bucle simple solo sirve para mantener la ventana abierta y detectar la tecla 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()