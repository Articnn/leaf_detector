import re

# Palabras clave y respuestas por clase
respuestas_por_clase = {
    "class_1": {
        "enfermedad": "En class_1 se recomienda eliminar hojas infectadas y aplicar fungicida preventivo.",
        "podar": "Podar zonas inferiores ayuda a que circule mejor el aire en class_1.",
        "propagación": "Evita mojar el follaje en class_1 para prevenir la propagación.",
        "daño": "Los daños en class_1 suelen ser visibles en los bordes. Retira esas hojas."
    },
    "class_2": {
        "enfermedad": "Usa extractos naturales como ajo o neem para tratar class_2.",
        "podar": "Podar ramas densas mejora el flujo de aire en plantas con class_2.",
        "propagación": "Aísla hojas infectadas de otras ramas si es class_2.",
        "daño": "Los daños por class_2 se extienden rápido. Podar es vital."
    },
    "class_3": {
        "enfermedad": "En class_3 se deben eliminar las hojas dañadas y mantener el suelo seco y bien drenado.",
        "podar": "Podar hojas inferiores favorece la aireación en class_3.",
        "propagación": "Evita acumulación de humedad, eso ayuda a reducir propagación en class_3.",
        "daño": "Los daños en class_3 son visibles como manchas oscuras. Elimina hojas afectadas."
    },
    "class_4": {
        "enfermedad": "Aplica fungicidas sistémicos en class_4 si las manchas avanzan.",
        "podar": "Desinfecta tus herramientas antes y después de podar en class_4.",
        "propagación": "Evita fertilizantes ricos en nitrógeno que favorecen la propagación en class_4.",
        "daño": "Los bordes amarillos y manchas son señales comunes de daño en class_4."
    },
    "class_5": {
        "enfermedad": "Controla class_5 con productos biológicos y reduce riego por aspersión.",
        "podar": "Retira hojas mal formadas y ramas secas para evitar acumulación de humedad.",
        "propagación": "Aumenta ventilación y espacio entre plantas para frenar propagación en class_5.",
        "daño": "Class_5 causa deformaciones y zonas necróticas. Corta las partes afectadas."
    },
    "class_6": {
        "enfermedad": "En class_6 es útil aplicar extractos de ajo o fungicidas naturales.",
        "podar": "Poda las zonas densas para evitar que la humedad se acumule en class_6.",
        "propagación": "Aísla las plantas afectadas y evita riego directo al follaje.",
        "daño": "Se observan manchas circulares en hojas en class_6. Remueve las más afectadas."
    },
    "class_7": {
        "enfermedad": "Aplica fungicida directamente al suelo si detectas síntomas en class_7.",
        "podar": "Poda hojas externas y mejora la circulación del aire en class_7.",
        "propagación": "El riego por goteo ayuda a evitar que se propague class_7.",
        "daño": "Manchas que aumentan de tamaño son comunes en class_7. Elimina esas hojas rápido."
    }
}


palabras_clave = ["enfermedad", "propagación", "podar", "daño"]

def detectar_nombre(mensaje):
    match = re.search(r"(me llamo|mi nombre es)\s+([A-Za-záéíóúñÑ]+)", mensaje.lower())
    if match:
        return match.group(2).capitalize()
    return None

def detectar_clase(mensaje):
    match = re.search(r"clase\s*_?(\d+)", mensaje.lower())
    if match:
        return f"class_{match.group(1)}"
    return None

def responder_por_tema(mensaje, clase_actual, nombre):
    for clave in palabras_clave:
        if clave in mensaje.lower():
            if clase_actual is None:
                return f"{nombre}, primero analiza una hoja para que pueda darte respuestas específicas."
            respuesta = respuestas_por_clase.get(clase_actual, {}).get(clave)
            if respuesta:
                return f"{nombre}, {respuesta}"
            else:
                return f"{nombre}, no tengo una respuesta específica para '{clave}' en {clase_actual}."
    return f"{nombre}, no entendí tu mensaje. Puedes preguntarme sobre enfermedades, podar, o clase de tu hoja."

def procesar_mensaje(user_input, estado):
    nombre = estado.get("nombre")
    clase_actual = estado.get("clase_actual", None)

    # Detectar si se dio el nombre
    nombre_detectado = detectar_nombre(user_input)
    if nombre_detectado:
        estado["nombre"] = nombre_detectado
        return f"Encantado, {nombre_detectado}. Ahora puedes preguntarme sobre tu planta."

    # Detectar clase (si el usuario la menciona)
    clase_detectada = detectar_clase(user_input)
    if clase_detectada:
        estado["clase_actual"] = clase_detectada
        clase_actual = clase_detectada

    # Si aún no hay nombre
    if not nombre:
        return "🌿 Hola, soy GreenBot 🌿. ¿Cómo gustas que te llame?\nEscribe: 'me llamo [tu nombre]'"

    # Procesar pregunta por tema
    return responder_por_tema(user_input, clase_actual, nombre)
