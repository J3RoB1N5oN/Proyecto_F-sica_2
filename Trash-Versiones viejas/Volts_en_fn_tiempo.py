import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import re # Para limpiar el texto de OCR

# --- 1. CONFIGURACIÓN INICIAL (¡AJUSTAR ESTO!) ---
# Apuntar esta variable a donde se instaló Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- RUTA AL VIDEO ORIGINAL ---
VIDEO_PATH_ORIGINAL = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4'

# --- ROI (Región de Interés) para el VOLTÍMETRO ---
VOLTIMETRO_ROI = (609, 271, 783, 345)

# Probar con el motor MODERNO (oem 3) y psm 6 (bloque de texto)
OCR_CONFIG = "-l eng --oem 3 --psm 6 -c tessedit_char_whitelist=0123456789."

# --- Lista para guardar los datos ---
datos_voltaje = []

# --- Función para pre-procesar la imagen para OCR ---
def preprocesar_para_ocr(img_roi):
    """
    Prepara la ROI para que Tesseract la lea mejor.
    """
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar un umbral (threshold)
    valor_umbral = 30
    
    # Opción A: Dígitos claros sobre fondo oscuro
    # (val, thresh) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Opción B: Dígitos oscuros sobre fondo claro
    (val, thresh) = cv2.threshold(gray, valor_umbral, 255, cv2.THRESH_BINARY_INV)
    
    final_img = thresh
    
    # 4. (Opcional) Mostrar la imagen pre-procesada para debug
    cv2.imshow("Debug OCR", final_img)
    cv2.waitKey(1)
    
    return final_img

# --- Función para limpiar la salida de OCR ---
def limpiar_salida_ocr(texto_ocr):
    """
    Limpia la salida de OCR y devuelve un float.
    """
    if not texto_ocr:
        return np.nan
        
    texto_limpio = re.sub(r"[^0-9.]", "", texto_ocr)
    
    try:
        valor = float(texto_limpio)
        return valor
    except ValueError:
        return np.nan

# --- 2. PROCESAMIENTO DEL VIDEO ---

print(f"Iniciando análisis de voltaje de: {VIDEO_PATH_ORIGINAL}")
cap = cv2.VideoCapture(VIDEO_PATH_ORIGINAL)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Total de frames: {total_frames}, FPS: {fps:.2f}, Dimensiones: {width}x{height}")


frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_time = frame_num / fps

    # Extraer la ROI del frame actual
    v_roi_img = frame[VOLTIMETRO_ROI[1]:VOLTIMETRO_ROI[3], VOLTIMETRO_ROI[0]:VOLTIMETRO_ROI[2]]

    # Pre-procesar la ROI
    v_roi_procesada = preprocesar_para_ocr(v_roi_img)
    
    # Aplicar OCR a la ROI
    try:
        texto_voltaje = pytesseract.image_to_string(v_roi_procesada, config=OCR_CONFIG)
    except Exception as e:
        print(f"\n¡ERROR DE TESSERACT! ¿Está instalado y en el PATH?")
        print(f"Error: {e}")
        print("Saliendo.")
        break

    # Limpiar el resultado
    voltaje = limpiar_salida_ocr(texto_voltaje)
    
    # Guardar los datos
    datos_voltaje.append({ # <--- Modificado
        "Tiempo_s": frame_time,
        "Voltaje_V": voltaje
    })
    
    # Mostrar progreso
    if frame_num % int(fps) == 0: # Cada segundo de video
        print(f"Procesando: Segundo {frame_time:.0f} | Leído: V={voltaje}") # <--- Modificado

    frame_num += 1
    
    # (Opcional) Mostrar el video con la ROI para debug
    cv2.rectangle(frame, (VOLTIMETRO_ROI[0], VOLTIMETRO_ROI[1]), (VOLTIMETRO_ROI[2], VOLTIMETRO_ROI[3]), (0, 255, 0), 2)
    cv2.imshow("Video Original", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- Análisis de video completado. ---")

# --- 3. PROCESAMIENTO CON PANDAS Y CSV ---

print("Creando DataFrame de Pandas...")
df = pd.DataFrame(datos_voltaje) 

# --- ¡NUEVO CHECK DE SEGURIDAD! ---
# Verificamos si el DataFrame está vacío ANTES de hacer nada.
if df.empty:
    print("\n---------------------------------------------------------")
    print("¡ERROR FATAL! No se pudo leer ningún dato del video.")
    print("La lista de datos 'datos_voltaje' está vacía.")
    print("\nCausa probable:")
    print("  La configuración de Tesseract ('OCR_CONFIG') está fallando")
    print("  en el primer frame y está forzando la salida del bucle.")
    print("  Prueba con otra configuración de 'oem' o 'psm'.")
    print("---------------------------------------------------------")
else:
    # --- SI NO ESTÁ VACÍO, PROCESAMOS NORMALMENTE ---
    print(f"Datos cargados ({len(df)} filas). Rellenando valores NaN (si los hay)...")
    
    # Rellenar valores NaN (donde OCR falló) - Sintaxis moderna
    df['Voltaje_V'] = df['Voltaje_V'].ffill()

    # Guardar en archivo CSV
    output_csv = "analisis_voltaje.csv" 
    df.to_csv(output_csv, index=False)
    print(f"¡Datos guardados exitosamente en '{output_csv}'!")

    # --- 4. GRÁFICO CON MATPLOTLIB ---
    print("Generando gráfico de resultados...")
    fig, ax = plt.subplots(figsize=(12, 6)) # Solo 1 gráfico

    # Gráfico de Voltaje
    ax.plot(df['Tiempo_s'], df['Voltaje_V'], label='Voltaje', color='blue')
    ax.set_ylabel('Voltaje (V)')
    ax.set_title('Análisis de Voltaje vs. Tiempo')
    ax.set_xlabel('Tiempo (s)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    output_png = "grafico_voltaje.png" 
    plt.savefig(output_png)
    print(f"¡Gráfico guardado exitosamente en '{output_png}'!")

    plt.show()

# "Proceso finalizado" se imprimirá en cualquier caso
print("--- Proceso finalizado. ---")