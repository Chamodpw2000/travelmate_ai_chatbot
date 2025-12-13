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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate critical environment variables
if not MONGODB_URI:
    print("WARNING: MONGODB_URI not set!")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Lazy load heavy resources
hf_model = None
mongo_client = None
tavily = None
llm = None
graph = None

def initialize_models():
    """Initialize models and connections on first use"""
    global hf_model, mongo_client, tavily, llm, graph
    
    if hf_model is None:
        print("Loading HuggingFace model...")
        try:
            hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("HuggingFace model loaded successfully")
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
    
    if mongo_client is None and MONGODB_URI:
        print("Connecting to MongoDB...")
        try:
            from pymongo import MongoClient
            mongo_client = MongoClient(MONGODB_URI, maxPoolSize=10, minPoolSize=1, serverSelectionTimeoutMS=5000)
            print("MongoDB connected successfully")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
    
    if llm is None:
        print("Initializing LLM...")
        try:
            llm = ChatOpenAI()
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
    
    if tavily is None:
        print("Initializing Tavily...")
        try:
            tavily = TavilySearchResults()
            print("Tavily initialized successfully")
        except Exception as e:
            print(f"Error initializing Tavily: {e}")
    
    if graph is None:
        print("Building LangGraph...")
        try:
            build_graph()
            print("LangGraph built successfully")
        except Exception as e:
            print(f"Error building LangGraph: {e}")

print("Flask app initialized - models will load on first request")

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
        if mongo_client is None or hf_model is None:
            initialize_models()
        
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

# Build graph function
def build_graph():
    """Build the LangGraph workflow"""
    global graph, llm, tavily
    
    # Initialize tavily if not done
    if tavily is None:
        tavily = TavilySearchResults()
    
    # Prepare tools
    tools = [search_destinations, tavily]
    
    # Initialize the LLM if not done
    if llm is None:
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
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("tool_calling_llm", tool_calling_llm)
    graph_builder.add_node("tools", ToolNode(tools))
    
    graph_builder.add_edge(START, "tool_calling_llm")
    graph_builder.add_conditional_edges("tool_calling_llm", tools_condition)
    graph_builder.add_edge("tools", "tool_calling_llm")
    
    # Compile the graph
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# Initialize memory globally for context API
memory = MemorySaver()

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
        # Initialize models on first request
        if graph is None:
            initialize_models()
        
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

@app.route('/query-context', methods=['POST'])
def query_context():
    """
    Extract answers from provided context data using OpenAI with conversation memory
    Expected JSON format:
    {
        "context": "Accommodations",
        "contextData": [
            {"name": "Hotel A", "location": "Colombo", "rating": 4.5, "price": 100},
            {"name": "Hotel B", "location": "Kandy", "rating": 4.0, "price": 80}
        ],
        "question": "What are the available hotels in the platform?",
        "thread_id": "user_123"  // Optional, defaults to "default_context"
    }
    """
    try:
        # Initialize LLM if needed
        if llm is None:
            initialize_models()
        
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        if 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        if 'contextData' not in data:
            return jsonify({"error": "Context data is required"}), 400
        
        question = data['question']
        context_data = data['contextData']
        context_type = data.get('context', 'general')
        thread_id = data.get('thread_id', 'default_context')
        
        # Detect if question is asking about a different context
        question_lower = question.lower()
        context_keywords = {
            'Accommodations': ['hotel', 'accommodation', 'stay', 'room', 'lodge', 'resort', 'villa', 'guesthouse'],
            'Restaurants': ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'cafe', 'eatery'],
            'Guides': ['guide', 'tour guide', 'tourist guide', 'travel guide'],
            'Destinations': ['destination', 'place', 'location', 'attraction', 'site', 'visit', 'tourist spot'],
            'Transportation': ['transport', 'transportation', 'vehicle', 'car', 'bus', 'train', 'taxi', 'ride', 'driver', 'rental', 'hire']
        }
        
        # Check if question mentions a different context
        detected_contexts = []
        for ctx, keywords in context_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_contexts.append(ctx)
        
        # If question clearly asks about a different context, suggest switching
        if detected_contexts and context_type not in detected_contexts and len(detected_contexts) == 1:
            suggested_context = detected_contexts[0]
            return jsonify({
                "response": f"It looks like you're asking about {suggested_context}, but the current context is set to {context_type}. Please switch to the correct context ({suggested_context}) to get accurate information about that data.",
                "context": context_type,
                "suggested_context": suggested_context,
                "status": "context_mismatch",
                "thread_id": thread_id
            }), 200
        
        # Convert context data to string format for better readability
        import json
        context_str = json.dumps(context_data, indent=2)
        
        # Escape curly braces to prevent template variable errors
        # Replace { with {{ and } with }} for proper escaping
        context_str_escaped = context_str.replace('{', '{{').replace('}', '}}')
        
        # Create a context-aware prompt
        system_message = f"""You are a helpful assistant for a Sri Lankan tourism platform called TravelMate. 
You have been provided with information about {context_type} available on the platform.
Your task is to answer user questions based on the provided data.

**Response Guidelines:**
- Answer in a clear, natural conversational manner
- For list queries: Provide an overview with key highlights of each item
- For specific queries: Focus on relevant details that answer the question
- Format as readable passages, never raw JSON or data dumps
- Mention key details: names, locations, prices, ratings, contact info
- Use proper formatting: organize information logically
- If asking about availability, mention available status when relevant
- If asking about prices, clearly state the price range or per-person cost
- If asking about ratings, mention the rating out of 5
- For contact queries, provide contactNumber, email, or website if available
- If data is insufficient to answer, politely state what's missing
- Keep responses concise but informative (2-4 sentences per item for lists)
- Always be supportive and positive about Sri Lankan tourism
- Use proper Sri Lankan place names and context
- Remember previous questions in this conversation and provide contextual follow-up answers
- Work with whatever data structure is provided - adapt to available fields

**Available Data for {context_type}:**
{context_str_escaped}
"""
        
        # Create prompt with conversation memory
        context_prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": system_message},
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create a simple chain with memory
        context_chain = context_prompt | llm
        
        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create messages list with the current question
        messages = [HumanMessage(content=question)]
        
        # Get conversation history from memory if exists
        try:
            checkpoint = memory.get(config)
            if checkpoint and 'messages' in checkpoint:
                # Prepend history to current messages
                messages = checkpoint['messages'] + messages
        except:
            pass  # No history exists yet
        
        # Invoke the chain
        response = context_chain.invoke({"messages": messages})
        
        # Save to memory
        try:
            updated_messages = messages + [AIMessage(content=response.content)]
            memory.put(config, {"messages": updated_messages})
        except:
            pass  # Memory save failed, continue anyway
        
        return jsonify({
            "response": response.content,
            "context": context_type,
            "status": "success",
            "thread_id": thread_id
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
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
