import cv2
import mediapipe as mp

# Inicializar el detector de manos
mp_deteccion_manos = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar el dibujado de manos
mp_dibujado_manos = mp.solutions.drawing_utils

# Función para detectar las manos y dibujar un rectángulo alrededor de ellas
def detectar_manos(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultado = mp_deteccion_manos.process(imagen)

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mp_dibujado_manos.draw_landmarks(
                imagen, mano, mp.solutions.hands.HAND_CONNECTIONS)
            dedos = obtener_dedos(mano, imagen.shape[1], imagen.shape[0])
            mostrar_dedos(imagen, dedos)

    return imagen

# Función para obtener los dedos de la mano
def obtener_dedos(mano, ancho, alto):
    dedos = {}
    # Diccionario de los puntos de referencia de los dedos
    # El valor del diccionario es True si el dedo está doblado y False si no lo está
    puntos_dedos = {
        4: (True, "pulgar"),
        8: (True, "indice"),
        12: (True, "corazon"),
        16: (True, "anular"),
        20: (True, "menique")
    }
    # Comprobar si el dedo está doblado
    for id_dedo, punto in enumerate(mano.landmark):
        x, y = int(punto.x * ancho), int(punto.y * alto)
        if id_dedo in puntos_dedos:
            dedos[puntos_dedos[id_dedo][1]] = punto.z > mano.landmark[id_dedo - 2].z
    return dedos

# Función para mostrar los dedos doblados
def mostrar_dedos(imagen, dedos):
    # Dibujar un rectángulo con el estado de los dedos
    x, y, w, h = 20, 20, 200, 200
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 0), -1)
    for i, (nombre, doblado) in enumerate(dedos.items()):
        color = (0, 255, 0) if doblado else (0, 0, 255)
        cv2.putText(imagen, f"{nombre}: {int(doblado)}", (x + 10, y + 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Bucle principal para capturar y mostrar la imagen en tiempo real
while True:
    ret, imagen = cap.read()
    if ret == False:
        break
    imagen = detectar_manos(imagen)
    cv2.imshow("Detección de manos", imagen)
    if cv2.waitKey(1) == 27:
        break

#