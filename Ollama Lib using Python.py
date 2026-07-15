import ollama
 
 
res = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "Hey mate, tell me 3 facts about JORDAN."}]) 
print(res)
print(res.get("message").get("content"))