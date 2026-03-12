from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)


"""
Usted es un calificador que evalua si la generación de un LLM se fundamenta en un conjunto de datos recuperados. 
Otorgue una puntuación binaria de "sí" o "no". "Sí" significa que la respuesta se fundamenta en el conjunto de datos.
"""
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# RunnableSequence: Sequence of `Runnable` objects, where the output of one is the input of the next
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
