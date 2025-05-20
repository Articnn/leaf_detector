import re

# === RESPUESTAS POR CLASE Y PALABRA CLAVE ===
respuestas_por_clase = {
    "class_1": {
        "enfermedad": "Se recomienda eliminar hojas infectadas y aplicar fungicida preventivo.",
        "podar": "Podar zonas inferiores ayuda a que circule mejor el aire.",
        "propagación": "Evita mojar el follaje para prevenir propagación.",
        "daño": "Los daños suelen verse en los bordes. Retira esas hojas."
    },
    "class_2": {
        "enfermedad": "Usa extractos naturales como ajo o neem para tratarla.",
        "podar": "Podar ramas densas mejora el flujo de aire.",
        "propagación": "Aísla hojas infectadas para evitar que se extienda.",
        "daño": "Actúa rápido. Los daños se expanden velozmente."
    },
    "class_3": {
        "enfermedad": "Retira hojas dañadas y asegúrate de un buen drenaje.",
        "podar": "Elimina hojas muertas o con manchas tempranas.",
        "propagación": "Evita humedad excesiva. Usa fungicidas si es necesario.",
        "daño": "Identifica manchas marrones o textura débil en la hoja."
    },
    "class_4": {
        "enfermedad": "Evita el exceso de nitrógeno. Aplica controles biológicos.",
        "podar": "Desinfecta herramientas antes de podar para evitar infecciones.",
        "propagación": "Evita mojar el follaje y aplica fungicidas preventivos.",
        "daño": "Busca manchas irregulares o textura rugosa."
    },
    "class_5": {
        "enfermedad": "Aplica control biológico y mantén humedad estable.",
        "podar": "Corta hojas deformadas o con puntos visibles.",
        "propagación": "Evita hacinamiento de hojas. Permite ventilación.",
        "daño": "Retira zonas amarillentas o con hongos visibles."
    },
    "class_6": {
        "enfermedad": "Usa extracto de ajo o fungicidas suaves. Aísla ramas afectadas.",
        "podar": "Poda las zonas densas para mejorar ventilación.",
        "propagación": "Mantén el área ventilada y sin agua estancada.",
        "daño": "Busca manchas redondas o color marrón en los bordes."
    },
    "class_7": {
        "enfermedad": "Riega directo al suelo. Aplica fungicida si hay manchas.",
        "podar": "Elimina hojas con puntas secas o retorcidas.",
        "propagación": "Evita regar en exceso. Mejora el drenaje.",
        "daño": "Observa bordes quebradizos o zonas con color extraño."
    },
}

palabras_clave = ["enfermedad", "propagación", "podar", "daño"]

# === FUNCIONES DE DETECCIÓN ===
def detectar_nombre(mensaje):
    match = re.search(r"(me llamo|mi nombre es)\s+([A-Za-záéíóúñÑ]+)", mensaje.lower())
    return match.group(2).capitalize() if match else None

def detectar_clase(mensaje):
    match = re.search(r"clase\s*_?(\d+)", mensaje.lower())
    return f"class_{match.group(1)}" if match else None

def detectar_palabra_clave(mensaje):
    for palabra in palabras_clave:
        if palabra in mensaje.lower():
            return palabra
    return None

# === INTENCIONES BÁSICAS CONVERSACIONALES ===
def detectar_intencion_basica(mensaje):
    mensaje = mensaje.lower()
    if any(p in mensaje for p in ["hola", "buenas"]):
        return "saludo"
    if any(p in mensaje for p in ["ayuda", "ocupo", "necesito"]):
        return "ayuda"
    if any(p in mensaje for p in ["gracias", "muy amable"]):
        return "agradecimiento"
    return None

# === RESPONDER ===
def responder_por_tema(mensaje, clase_actual, nombre):
    tema = detectar_palabra_clave(mensaje)
    if not tema:
        return f"{nombre}, no entendí tu mensaje. Puedes preguntarme sobre enfermedades, poda, propagación o daño."

    if clase_actual is None:
        return f"{nombre}, primero analiza una hoja o menciona la clase para darte una recomendación precisa."

    respuesta = respuestas_por_clase.get(clase_actual, {}).get(tema)
    if respuesta:
        return f"{nombre}, {respuesta}"
    else:
        return f"{nombre}, no tengo una respuesta específica para '{tema}' en {clase_actual}."

# === FLUJO PRINCIPAL ===
def procesar_mensaje(user_input, estado):
    nombre = estado.get("nombre")
    clase_actual = estado.get("clase_actual")

    # Intento de saludo/conversación
    intencion = detectar_intencion_basica(user_input)
    if intencion == "saludo":
        return "🌿 Hola, soy GreenBot 🌿. ¿Cómo gustas que te llame?\nEscribe: 'me llamo [tu nombre]'"
    elif intencion == "ayuda":
        return "Puedes preguntarme sobre cómo tratar enfermedades, poda, propagación o tipo de daño."
    elif intencion == "agradecimiento":
        return f"{nombre or 'Amigo'}, ¡Con gusto! Si necesitas algo más, estoy aquí."

    # Guardar nombre
    nombre_detectado = detectar_nombre(user_input)
    if nombre_detectado:
        estado["nombre"] = nombre_detectado
        return f"Encantado, {nombre_detectado}. Ahora puedes preguntarme sobre tu planta."

    # Detectar clase en texto
    clase_detectada = detectar_clase(user_input)
    if clase_detectada:
        estado["clase_actual"] = clase_detectada
        clase_actual = clase_detectada

    # Si aún no hay nombre
    if not nombre:
        return "🌿 Hola, soy GreenBot 🌿. ¿Cómo gustas que te llame?\nEscribe: 'me llamo [tu nombre]'"

    return responder_por_tema(user_input, clase_actual, nombre)
