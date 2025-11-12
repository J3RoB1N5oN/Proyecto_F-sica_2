import cv2
import numpy as np
import math
import csv

# --- CONFIGURACIÓN ---
video_path = 'WhatsApp Video 2025-09-29 at 20.17.52.mp4'
area_minima = 500
DURACION_MAXIMA_SEGUNDOS = 60

# --- ESCALA ---
escala_metros_por_pixel = 0.0005 # ¡RECUERDA CAMBIAR ESTE VALOR!

# --- NOMBRE DEL ARCHIVO DE SALIDA ---
nombre_archivo_csv = 'resultados_analisis_fisica.csv'


# --- INICIALIZACIÓN ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
max_fotogramas = int(DURACION_MAXIMA_SEGUNDOS * fps)
print(f"Analizando video a {fps:.2f} FPS. El tiempo entre fotogramas (Δt) es {1/fps:.4f} segundos.")

background_subtractor = cv2.createBackgroundSubtractorMOG2()

puntos_trayectoria = []

# --- PROCESAMIENTO DEL VIDEO ---
while True:
    fotograma_actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()

    if not ret or fotograma_actual >= max_fotogramas:
        if not ret: print("Fin del video.")
        else: print(f"Análisis detenido al alcanzar el límite de {DURACION_MAXIMA_SEGUNDOS} segundos.")
        break

    foreground_mask = background_subtractor.apply(frame)
    
    # Se mantiene la corrección anterior para compatibilidad
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, 2)

    for contour in contours:
        if cv2.contourArea(contour) > area_minima:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            centro_x = int(x + w / 2)
            centro_y = int(y + h / 2)
            puntos_trayectoria.append((centro_x, centro_y))

    for i in range(1, len(puntos_trayectoria)):
        cv2.line(frame, puntos_trayectoria[i - 1], puntos_trayectoria[i], (0, 0, 255), 2)

    cv2.imshow('Análisis de Movimiento y Trayectoria', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# --- FINALIZACIÓN Y ANÁLISIS ---
cap.release()
cv2.destroyAllWindows()

# --- CÁLCULO DE VELOCIDAD Y EXPORTACIÓN DE DATOS A CSV ---
print("\n--- INICIANDO ANÁLISIS FINAL Y EXPORTACIÓN DE DATOS ---")

if len(puntos_trayectoria) > 1:
    delta_t = 1.0 / fps
    with open(nombre_archivo_csv, 'w', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow([
            'Tiempo (s)', 'Fotograma', 'Posicion X (px)', 'Posicion Y (px)',
            'Velocidad X (m/s)', 'Velocidad Y (m/s)', 'Rapidez (m/s)'
        ])

        for i in range(1, len(puntos_trayectoria)):
            punto_anterior = puntos_trayectoria[i - 1]
            punto_actual = puntos_trayectoria[i]
            
            delta_x_pixeles = punto_actual[0] - punto_anterior[0]
            delta_y_pixeles = punto_actual[1] - punto_anterior[1]
            
            delta_x_metros = delta_x_pixeles * escala_metros_por_pixel
            delta_y_metros = delta_y_pixeles * escala_metros_por_pixel
            
            velocidad_x_mps = delta_x_metros / delta_t
            velocidad_y_mps = delta_y_metros / delta_t
            
            rapidez_mps = math.sqrt(velocidad_x_mps**2 + velocidad_y_mps**2)
            
            tiempo_actual = i * delta_t
            
            fila_de_datos = [
                f"{tiempo_actual:.4f}", i, punto_actual[0], punto_actual[1],
                f"{velocidad_x_mps:.4f}", f"{velocidad_y_mps:.4f}", f"{rapidez_mps:.4f}"
            ]
            escritor_csv.writerow(fila_de_datos)
    
    print(f"\n✅ ¡Éxito! Los datos han sido guardados en el archivo '{nombre_archivo_csv}'")

else:
    print("No se recolectaron suficientes puntos para calcular la velocidad y exportar los datos.")