from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import warnings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js backend communication

# Get environment variables from .env
MONGODB_URI = os.getenv("MONGODB_URI")

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load HuggingFace model
print("Loading HuggingFace model...")
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create a single MongoDB client to reuse
from pymongo import MongoClient
mongo_client = MongoClient(MONGODB_URI, maxPoolSize=10, minPoolSize=1, serverSelectionTimeoutMS=5000)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define search_destinations tool
@tool
def search_destinations(query_text: str) -> str:
    """
    Retrieve information using vector search to answer a user query.
    """
    try:
        collection = mongo_client["destinations_db"]["destinations"]

        # Fetch all documents with embeddings
        docs = list(collection.find({"hf_embedding": {"$exists": True}}))
        
        if not docs:
            return "No destination information available in the database."
        
        query_embedding = hf_model.encode(query_text)

        # Compute similarity scores
        results = []
        for doc in docs:
            emb = doc.get("hf_embedding")
            if emb:
                score = cosine_similarity(np.array(query_embedding), np.array(emb))
                results.append((score, doc))

        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Get top k results
        top_k = 3
        top_results_formatted = []
        for score, doc in results[:top_k]:
            top_results_formatted.append(f"Name: {doc.get('name', 'Unknown')}, Description: {doc.get('description', 'No description')}")

        return "\n".join(top_results_formatted) if top_results_formatted else "No matching destinations found."
    except Exception as e:
        return f"Error searching destinations: {str(e)}"

# Tavily Search Tool
tavily = TavilySearchResults()

# Prepare tools
tools = [search_destinations, tavily]

# Initialize the LLM
llm = ChatOpenAI()

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
        <role>
        You are a helpfull chatbot in a Sri Lankan based trip planning and travelling application. Users can book trips, explore information about
        travelling desntinations in sri lank, explore hotels, restaurants, villas and tourist guides in sri lanka. Your task is to provide meaning full answers
        for the questions asked by users. To answer question you can use your knowledge about that particular location, or if you need some specific information
        you have been provided some tools as well. You can use those tools and provide an meaning full answer to the user.
        <role/>

        <instruction>
        Follow the below instructions when you are ansering an question.
        * Your answer should always supportive to the sri lanka tourism industry. Never provide any negative answer.
        * If the user's question is not related to Sri Lankan tourism, politely inform them that you can only assist with tourism-related questions about Sri Lanka.
        * If the tools provide insufficient information, you can use your own knowledge to answer the question.
        * When providing the answer always give the answer as a passage. Never provide unnecessary tags or formats. Just provide as a passage.
        * The message should me neither too long nor too short. Always provide a medium sized answer that is easy to read.
        </instruction>

        #Guadrails
        * Always answer to the sri lanka tourism related questions. Newver divert from the main topic.
        """
    },
    MessagesPlaceholder(variable_name="messages"),
])

prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

# Prepare the LLM with tools
bind_tools = llm.bind_tools(tools)
llm_with_tools = prompt | bind_tools

# Define the graph state
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

# Node definition
def tool_calling_llm(state: GraphState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Initialize memory
memory = MemorySaver()

# Build the graph
print("Building LangGraph...")
graph_builder = StateGraph(GraphState)
graph_builder.add_node("tool_calling_llm", tool_calling_llm)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "tool_calling_llm")
graph_builder.add_conditional_edges("tool_calling_llm", tools_condition)
graph_builder.add_edge("tools", "tool_calling_llm")

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)

print("Flask app is ready!")

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Chatbot is running"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Expected JSON format:
    {
        "message": "Tell me about Colombo",
        "thread_id": "user_123"  // Optional, defaults to "default"
    }
    """
    try:
        data = request.json
        
        # Validate input
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        user_message = data['message']
        thread_id = data.get('thread_id', 'default')
        
        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the graph
        response = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )
        
        # Extract the AI's response (last message)
        ai_message = None
        for msg in reversed(response['messages']):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        return jsonify({
            "response": ai_message,
            "thread_id": thread_id,
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """
    Reset conversation for a specific thread
    Expected JSON format:
    {
        "thread_id": "user_123"
    }
    """
    try:
        data = request.json
        thread_id = data.get('thread_id', 'default')
        
        # Note: MemorySaver doesn't have a direct reset method
        # In production, you might want to use a different checkpointer
        # For now, we'll just acknowledge the request
        
        return jsonify({
            "message": f"Conversation reset requested for thread: {thread_id}",
            "status": "success",
            "note": "Memory will be fresh with new thread_id"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Cleanup handler
import atexit

@atexit.register
def cleanup():
    """Close MongoDB connection on shutdown"""
    try:
        mongo_client.close()
        print("MongoDB connection closed.")
    except Exception:
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
