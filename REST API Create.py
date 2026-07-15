import requests
import json

url = "http://localhost:11434/api/create"

payload = {
    "name": "OMDEH",
    "from": "llama3.2",
    "system": "You are OMDEH, A professional Machine and Deep Learning Engineer. Your purpose is to assist beginners in AI by answering \
        thier questions and providing guidance. Make sure to provide clear and concise explanations, and offer practical examples when necessary. Always encourage learning and curiosity in the field of AI.",
}

print("Initiating model creation...")
# stream=True is critical here
response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            # Each line is a separate JSON object
            data = json.loads(line.decode('utf-8'))
            
            # Print the status update
            status = data.get("status", "")
            print(f"Status: {status}")
            
            # Check if creation is done
            if status == "success":
                print("\n✅ Model 'OMDEH' created successfully!")
                break
            elif "error" in data:
                print(f"\n❌ Error during creation: {data['error']}")
                break
else:
    print(f"Failed to connect: {response.status_code}")