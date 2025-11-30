from langchain_ollama import OllamaEmbeddings, OllamaLLM


def get_embedding():
    return OllamaEmbeddings(model="all-minilm:33m-l12-v2-fp16")


def get_llm():
    return OllamaLLM(model="llama3.2:3b-instruct-q8_0")
