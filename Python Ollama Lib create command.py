import ollama

modelfile_content = """ \
FROM llama3.2 \
SYSTEM You are a professional cybersecurity analyst. You have extensive knowledge of network security. \
PARAMETER temperature 0.7 \
"""

# Attempt the create call
ollama.create(model="OMDEH_CYBER", modelfile=modelfile_content)

res = ollama.generate(model="OMDEH_CYBER", prompt="What are the best practices for securing a corporate network against ransomware attacks?")

print(res["response"])