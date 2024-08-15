from flask import Flask, request, jsonify, render_template, redirect
import fitz  # PyMuPDF
import uuid
from collections import defaultdict
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import chromadb

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

# In-memory storage for chat sessions and history
chat_sessions = {}
chat_history = defaultdict(list)
pdf_contents = {}  # Store extracted text by asset_id

# Setup LangChain components
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=api_key, temperature=0.7)

# Define PromptTemplate
template = """
You are a fortune teller. These Human will ask you a question about their life. 
Use the following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# Define RAG Chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def get_context_from_collection(query):
    docs = collection.similarity_search(query)
    docs_text = "\n".join([doc['text'] for doc in docs]) if docs else ""
    return docs_text

rag_chain = (
    {"context": RunnablePassthrough(get_context_from_collection), "question": RunnablePassthrough()}
    | prompt 
    | llm
    | StrOutputParser()
)

class ChatBot:
    def __init__(self):
        self.rag_chain = rag_chain
 
    def chat(self, user_message):
        # Prepare the input data for the chain
        context = get_context_from_collection(user_message)
        input_data = {"context": context, "question": user_message}
        
        # Invoke the chain with the input data
        result = self.rag_chain.invoke(input=input_data)
        return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.txt'):
            text = file.read().decode('utf-8').strip()
            asset_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[asset_id])
            pdf_contents[asset_id] = text  # Store the text for interaction
            return jsonify({'asset_id': asset_id})

        elif file and file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
            asset_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[asset_id])
            pdf_contents[asset_id] = text  # Store the text for interaction
            return jsonify({'asset_id': asset_id})
        
    return render_template('index.html')

@app.route('/api/documents/process', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8').strip()
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    asset_id = str(uuid.uuid4())
    collection.add(documents=[text], ids=[asset_id])
    pdf_contents[asset_id] = text  # Store the text for interaction

    return jsonify({'asset_id': asset_id}), 200

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    data = request.json
    asset_id = data.get('asset_id')

    if not asset_id:
        return jsonify({'error': 'Asset ID is required'}), 400

    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = asset_id
    return jsonify({'chat_id': chat_id}), 200

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    data = request.json
    chat_id = data.get('chat_id')
    user_message = data.get('message')

    if not chat_id or not user_message:
        return jsonify({'error': 'Chat ID and message are required'}), 400

    if chat_id not in chat_sessions:
        return jsonify({'error': 'Invalid Chat ID'}), 404

    asset_id = chat_sessions[chat_id]
    pdf_text = pdf_contents.get(asset_id, "")

    try:
        # Use the ChatBot class
        bot = ChatBot()
        response_message = bot.chat(user_message)

        # Update chat history
        chat_history[chat_id].append({'user': user_message, 'bot': response_message})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

    return jsonify({'response': response_message}), 200

if __name__ == '__main__':
    bot = ChatBot()
    user_message = "How many awards did Nathan win?"
    result = bot.chat(user_message)
    print(result)

