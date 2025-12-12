from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import warnings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from abc import ABC, abstractmethod

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

# ============= MongoDB Query Agent Implementation =============

# Output Parser for MongoDB queries
class MongoDBQueryResponse(BaseModel):
    query: list = Field(description="MongoDB aggregation pipeline as a list")
    
    def to_dict(self):
        return self.query

mongodb_parser = PydanticOutputParser(pydantic_object=MongoDBQueryResponse)

# Database schemas and descriptions for TravelMate
COLLECTION_SCHEMAS = {
    "accommodations": {
        "schema": """{
  "name": "string",
  "address": "string",
  "contactNumber": "string",
  "description": "string",
  "category": "string",
  "distance_from_city": "string",
  "perPerson_price": "number",
  "available": "boolean",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the accommodation (hotel, villa, resort)
2. **address**: Physical address of the accommodation
3. **contactNumber**: Contact phone number
4. **description**: Detailed description of the accommodation
5. **category**: Type of accommodation (e.g., Hotel, Villa, Resort)
6. **distance_from_city**: Distance from the nearest city
7. **perPerson_price**: Price per person per night
8. **available**: Whether the accommodation is currently available
9. **rating**: Rating from 0 to 5
"""
    },
    "destinations": {
        "schema": """{
  "name": "string",
  "city": "string",
  "distanceFromColombo": "number",
  "category": "array of strings",
  "bestTimeToVisit": "string",
  "description": "string",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the destination
2. **city**: City where the destination is located
3. **distanceFromColombo**: Distance from Colombo in kilometers
4. **category**: Categories of the destination (e.g., Beach, Historical, Wildlife)
5. **bestTimeToVisit**: Best time/season to visit
6. **description**: Detailed description of the destination
7. **rating**: Rating from 0 to 5
"""
    },
    "guides": {
        "schema": """{
  "name": "string",
  "area": "array of strings",
  "languages": "array of strings",
  "chargesPerDay": "number",
  "description": "string",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the tour guide
2. **area**: Areas where the guide operates
3. **languages**: Languages spoken by the guide
4. **chargesPerDay**: Daily charges for hiring the guide
5. **description**: Description of guide's expertise and experience
6. **rating**: Rating from 0 to 5
"""
    },
    "restaurants": {
        "schema": """{
  "name": "string",
  "address": "string",
  "category": "array of strings",
  "mainCategory": "string",
  "contactNumber": "string",
  "description": "string",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the restaurant
2. **address**: Physical address
3. **category**: Food categories served (e.g., Italian, Chinese, Seafood)
4. **mainCategory**: Primary cuisine type
5. **contactNumber**: Contact phone number
6. **description**: Description of the restaurant
7. **rating**: Rating from 0 to 5
"""
    },
    "transportationservices": {
        "schema": """{
  "name": "string",
  "pricePerHour": "number",
  "address": "string",
  "contactNumber": "string",
  "description": "string",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the transportation service
2. **pricePerHour**: Hourly rental price
3. **address**: Service location address
4. **contactNumber**: Contact phone number
5. **description**: Description of vehicles and services
6. **rating**: Rating from 0 to 5
"""
    },
    "preplannedtrips": {
        "schema": """{
  "name": "string",
  "mainDestinations": "array of strings",
  "mainActivities": "array of strings",
  "price": "number",
  "duration": "number",
  "description": "string",
  "rating": "number"
}""",
        "description": """
1. **name**: Name of the pre-planned trip package
2. **mainDestinations**: Main destinations covered in the trip
3. **mainActivities**: Main activities included
4. **price**: Total price for the trip package
5. **duration**: Duration in days
6. **description**: Detailed description of the trip
7. **rating**: Rating from 0 to 5
"""
    }
}

# BaseChain class following the article pattern
class BaseChain(ABC):
    llm = None
    
    def __init__(self, model_name, temperature, **kwargs):
        self.llm = OpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            temperature=temperature,
        )
    
    @property
    @abstractmethod
    def template(self) -> str:
        """Template string for the prompt"""
        pass
    
    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """List of input variables"""
        pass
    
    @staticmethod
    @abstractmethod
    def populate_input_variables(payload: dict) -> dict:
        """Populate input variables from payload"""
        pass
    
    @abstractmethod
    def partial_variables(self) -> dict:
        """Partial variables for the chain"""
        pass
    
    @abstractmethod
    def chain(self) -> LLMChain:
        """Create and return the LLM chain"""
        pass

# MongoDB Query Agent
class MongoDBQueryAgent(BaseChain):
    def __init__(self, collection_name: str):
        super().__init__(
            model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo-instruct"),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.1")),
        )
        self.collection_name = collection_name
        self.collection_info = COLLECTION_SCHEMAS.get(collection_name, {})
    
    @property
    def template(self) -> str:
        return """Create a MongoDB aggregation query for the following user question:
    
###{user_message}###

This is the collection schema: {table_schema}

This is the schema description: {schema_description}

Important instructions:
- Return ONLY a MongoDB aggregation pipeline as a JSON array
- Use $match, $project, $sort, $limit stages as needed
- For text searches, use $regex with case-insensitive option
- For numeric comparisons, use appropriate operators ($gt, $lt, $gte, $lte, $eq)
- For rating queries, the rating field is a number from 0 to 5
- Sort results by relevance or rating when appropriate
- Limit results to top 5 most relevant items

{format_instructions}
"""
    
    @property
    def input_variables(self) -> List[str]:
        return ["user_message", "table_schema", "schema_description"]
    
    @staticmethod
    def populate_input_variables(payload: dict) -> dict:
        collection_name = payload.get("collection_name", "destinations")
        collection_info = COLLECTION_SCHEMAS.get(collection_name, {})
        
        return {
            "user_message": payload["question"],
            "table_schema": collection_info.get("schema", "{}"),
            "schema_description": collection_info.get("description", "")
        }
    
    def partial_variables(self) -> dict:
        return {"format_instructions": mongodb_parser.get_format_instructions()}
    
    def chain(self) -> LLMChain:
        chain_prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=self.template,
            partial_variables=self.partial_variables(),
        )
        query_builder_chain = LLMChain(
            llm=self.llm,
            prompt=chain_prompt,
        )
        return query_builder_chain

# Tool to search MongoDB using natural language
@tool
def search_mongodb(query_text: str, collection: str = "destinations") -> str:
    """
    Search TravelMate database using natural language queries. 
    Supports searching: destinations, accommodations, guides, restaurants, transportationservices, preplannedtrips.
    Examples: 
    - "Find beach destinations"
    - "Show me hotels under 5000 per person"
    - "Find guides who speak English"
    """
    try:
        # Validate collection name
        collection_lower = collection.lower()
        if collection_lower not in COLLECTION_SCHEMAS:
            return f"Invalid collection. Available: {', '.join(COLLECTION_SCHEMAS.keys())}"
        
        # Create the MongoDB query agent
        agent = MongoDBQueryAgent(collection_lower)
        query_chain = agent.chain()
        
        # Generate the MongoDB query
        chain_input = MongoDBQueryAgent.populate_input_variables({
            "question": query_text,
            "collection_name": collection_lower
        })
        
        query_response = query_chain.run(chain_input)
        
        # Parse the response to get the aggregation pipeline
        try:
            parsed_query = mongodb_parser.parse(query_response).to_dict()
        except Exception as parse_error:
            # If parsing fails, try to extract JSON array from response
            import json
            import re
            json_match = re.search(r'\[.*\]', query_response, re.DOTALL)
            if json_match:
                parsed_query = json.loads(json_match.group())
            else:
                return f"Could not generate valid MongoDB query. Error: {str(parse_error)}"
        
        # Execute the query on MongoDB
        db = mongo_client["destinations_db"]
        mongo_collection = db[collection_lower]
        
        results = list(mongo_collection.aggregate(parsed_query))
        
        if not results:
            return f"No {collection_lower} found matching your query."
        
        # Format results
        formatted_results = []
        for idx, doc in enumerate(results[:5], 1):
            result_str = f"{idx}. "
            if 'name' in doc:
                result_str += f"Name: {doc['name']}"
            if 'description' in doc:
                desc = doc['description'][:150] + "..." if len(doc.get('description', '')) > 150 else doc.get('description', '')
                result_str += f", Description: {desc}"
            if 'rating' in doc:
                result_str += f", Rating: {doc['rating']}/5"
            if 'price' in doc:
                result_str += f", Price: {doc['price']}"
            if 'perPerson_price' in doc:
                result_str += f", Price per person: {doc['perPerson_price']}"
            if 'chargesPerDay' in doc:
                result_str += f", Charges per day: {doc['chargesPerDay']}"
            
            formatted_results.append(result_str)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching MongoDB: {str(e)}"

# ============= End MongoDB Query Agent =============

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

# Prepare tools - now includes MongoDB query tool
tools = [search_destinations, search_mongodb, tavily]

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
        * You have access to search_mongodb tool which can query our database for destinations, accommodations, guides, restaurants, transportation services, and pre-planned trips.
        * Use search_mongodb tool when users ask about specific places, hotels, guides, restaurants, or services in Sri Lanka.
        * The search_destinations tool uses vector search and is good for general destination queries.
        </instruction>

        #Guadrails
        * Always answer to the sri lanka tourism related questions. Newver divert from the main topic.
        * When using search_mongodb, specify the correct collection: destinations, accommodations, guides, restaurants, transportationservices, or preplannedtrips.
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
