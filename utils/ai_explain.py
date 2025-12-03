import google.generativeai as genai
import  os


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_rag(query,documents):

    context_block = "\n\n".join(documents)

    prompt = f"""
    You are a wine expert AI.
    You can answer ONLY using the information from the context below.
    Do NOT hallucinate. If the answer is not in the documents, say "I don't know".

User query: {query}
    
    
Context from retrieved documents:
--------------------
{context_block}
--------------------

Provide a short but clear explanation.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    return response.text