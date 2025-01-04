just an attempt to try to create a local RAG system that can help easily understand complex documents

# Run:
1. Clone the code and install the dependencies specified in [requirements.txt](./requirements.txt)
2. Add a pdf that you want to _chat_ with
3. Have ollama running (preferably with the specified qwen model for the best results)
4. run `python conversations.py <path to pdf>`

## Specifications:

- embedding model: `mxbai-embed-large`
- inference model: `qwen2.5:7b-instruct-q8_0` (tried mixtral large and phi3 and llama3, at q4 and q8(wherever available). Qwen models felt like the best. )
- model server was run using `ollama`  (it is really hard to run these models via code and keep them dynamic during this POC stage, might pickup later)
- mostly built with `langchain` (no preference, just the first framework that got suggested)

### Features:

- keeps context of history of chat
- does not re-embed a document if they already exist

---

_requirements.txt generated using `pip3 freeze > requirements.txt`_