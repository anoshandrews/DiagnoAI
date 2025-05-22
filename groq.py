from groq import Groq
client = Groq()
models = client.models.list()
print(models)