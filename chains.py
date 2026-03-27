import datetime

from dotenv import load_dotenv

load_dotenv()


from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser, 
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer



llm = ChatOpenAI(model="o4-mini")
#parser = JsonOutputToolsParser(return_id=True)

# Crea un parser que toma la salida del LLM y la convierte automáticamente en una 
# instancia de AnswerQuestion. Si el LLM devuelve JSON que coincide con la 
# estructura de AnswerQuestion, este parser lo transforma en un objeto Python.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),  # MessagesPlaceholder: un marcador de posición que se reemplazará con el historial de mensajes (donde irá la pregunta del usuario)
        ("system", "Answer the user's question above using the required format."),  # Segundo mensaje del sistema: indica que responda en el formato requerido.
]).partial(time=lambda: datetime.datetime.now().isoformat(),)  # inyecta la hora actual automaticamente sin necesidad de pasarla en cada llamada


first_responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer.")  # Primer respondedor (respuesta inicial). Toma el template base y reemplaza {first_instruction} con una instruccion específica para la primera version: "Proporciona una respuesta detallada de ~250 palabras".

"""
1. Toma el prompt template:

2. Lo pasa al LLM configurado con tool calling:
 
    . tools=[AnswerQuestion]: el LLM puede usar la herramienta AnswerQuestion

    . tool_choice="AnswerQuestion": fuerza al LLM a usar esa herramienta específica

El resultado es que el LLM generará una respuesta estructurada que se ajusta al formato de AnswerQuestion.
"""
first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")


"""
Qué hace: Define las instrucciones para la fase de revisión, que requieren:
    Usar la crítica previa
    Añadir citas numéricas
    Incluir sección de referencias
    Eliminar información superflua
"""
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

"""
Qué hace: Similar al primer respondedor, pero:
    Usa las instrucciones de revisión
    Fuerza al LLM a usar la herramienta ReviseAnswer (que incluye el campo references)
"""
revisor = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    # Qué hace: Define la pregunta del usuario sobre SOC autónomo y startups que han levantado capital.
    # human_message = HumanMessage(
    #     content="Write about AI-Powered SOC / autonomous soc problem domain,"
    #     " list startups that do that and raised capital."
    # )
    
    # human_message = HumanMessage(
    #     content="Write about the problems of AI-powered booking engines / autonomous SOCs,"
    #     " list the startups that are doing this and have secured funding."
    # )
    
    """
    Escribe sobre los problemas de los motores de reservas de hoteles. Dime cuales son los principales 
    problemas que tienen las motores de reserva con respuecto a la busqueda 
    inicial de itinerarios y con respecto a las politicas de cancelaciones de cada reserva.
    """
    human_message = HumanMessage(
        content="Escribe sobre los problemas de los motores de reservas de hoteles."
        " Dime cuales son los principales problemas que tienen las motores de reserva con respecto a la busqueda"
        " inicial de itinerarios y con respecto a las politicas de cancelaciones de cada reserva."
    )

    """
    Qué hace: Construye la cadena final que se ejecutará:
        1. first_responder_prompt_template: toma el prompt con la pregunta
        2. llm.bind_tools(...): llama al LLM forzándolo a usar AnswerQuestion
        3. parser_pydantic: convierte la salida del LLM en un objeto AnswerQuestion
    
    """
    chain = (first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic)

    res = chain.invoke(input={"messages": [human_message]})
    
    print(res)
