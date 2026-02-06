"""
Omvyx Voice — FAQ Knowledge Base

Simple keyword-match FAQ lookup.  In production, swap this for a
vector-similarity search over an embedding index.
"""

from __future__ import annotations


_FAQ_ENTRIES: list[dict[str, str]] = [
    {
        "keywords": ["ubicación", "dirección", "dónde", "location", "where", "address"],
        "answer": (
            "Estamos ubicados en Calle Gran Vía 28, Madrid. "
            "El horario de atención es de lunes a viernes, de 9:00 a 17:00."
        ),
    },
    {
        "keywords": ["horario", "hora", "hours", "schedule", "abierto", "open"],
        "answer": (
            "Nuestro horario de atención es de lunes a viernes, "
            "de 9:00 a 17:00. Los fines de semana permanecemos cerrados."
        ),
    },
    {
        "keywords": ["precio", "costo", "tarifa", "price", "cost", "rate"],
        "answer": (
            "La consulta inicial tiene un costo de 50 €. "
            "Los precios de servicios adicionales dependen del tratamiento. "
            "¿Le gustaría agendar una cita para recibir un presupuesto personalizado?"
        ),
    },
    {
        "keywords": ["cancelar", "cancel", "anular"],
        "answer": (
            "Puede cancelar o reprogramar su cita con al menos 24 horas de antelación "
            "sin cargo alguno. Para cancelaciones con menos de 24 horas, "
            "se aplica un cargo del 50 %."
        ),
    },
    {
        "keywords": ["seguro", "insurance", "cobertura", "coverage"],
        "answer": (
            "Aceptamos los principales seguros médicos: Sanitas, Adeslas, Mapfre y DKV. "
            "Le recomendamos verificar su cobertura específica con su aseguradora."
        ),
    },
    {
        "keywords": ["parking", "estacionamiento", "aparcar"],
        "answer": (
            "Disponemos de parking gratuito para pacientes en el sótano del edificio. "
            "La entrada se encuentra en la calle lateral."
        ),
    },
]


async def search_faq(query: str) -> str | None:
    """
    Return the best matching FAQ answer, or None if no match.
    """
    query_lower = query.lower()
    for entry in _FAQ_ENTRIES:
        if any(kw in query_lower for kw in entry["keywords"]):
            return entry["answer"]
    return None
