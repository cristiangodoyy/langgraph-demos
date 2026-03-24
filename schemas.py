from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """ Define un modelo para la autocrítica. El LLM deberá identificar: """
    missing: str = Field(description="Critique of what is missing.")  # missing: qué información falta en la respuesta
    superfluous: str = Field(description="Critique of what is superfluous")  # superfluous: qué información sobra o es irrelevante


class AnswerQuestion(BaseModel):
    """Answer the question. Define la estructura que debe seguir la respuesta inicial. El LLM debe generar 3 cosas: """

    answer: str = Field(description="~250 word detailed answer to the question.")  # answer: la respuesta detallada (~250 palabras)
    reflection: Reflection = Field(description="Your reflection on the initial answer.")  # reflection: una autocrítica usando el modelo Reflection
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )  # search_queries: 1-3 consultas para investigar mejoras


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Se usa para la versión revisada, 
    donde el LLM debe incluir citas que respalden la informacion."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
