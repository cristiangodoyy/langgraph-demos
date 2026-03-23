import random


from dotenv import load_dotenv

load_dotenv()


from graph.graph import app


"""


# Define the list of strings
string_list = ["apple", "banana", "cherry", "date", "elderberry"]

# Use random.choice() to select a random string
random_string = random.choice(string_list)

# Print the result
print(random_string)
"""


if __name__ == '__main__':
    print('Hello advanced RAG')
    questions = ["agent memory?", "what is agent memory?", "What's the capital of Argentina?", "how to make pizza?"]
    question = random.choice(questions)
    #print(app.invoke(input={"question": "what is agent memory?"}))
    #print(app.invoke(input={"question": "What's the capital of Argentina?"}))
    print(app.invoke(input={"question": question}))
    #print(app.invoke(input={"question": "how to make pizza?"}))
