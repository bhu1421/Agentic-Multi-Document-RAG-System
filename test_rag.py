from backend.rag import get_agentic_response

print("Testing causal chat...")
response = get_agentic_response("Hi, how are you?")
print("Response:", response)

print("\nTesting document query...")
response = get_agentic_response("What is the main topic of the uploaded documents?")
print("Response:", response)
