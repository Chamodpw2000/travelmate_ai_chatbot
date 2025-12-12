# Flask Chatbot API Documentation

## Overview
This Flask API provides endpoints to communicate with the LangGraph-based Sri Lankan tourism chatbot from your Node.js backend.

## Installation

```bash
pip install flask flask-cors
```

## Running the Server

```bash
python flask_app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "message": "Chatbot is running"
}
```

### 2. Chat
**Endpoint:** `POST /chat`

**Request Body:**
```json
{
  "message": "Tell me about beaches in Sri Lanka",
  "thread_id": "user_123"
}
```

**Parameters:**
- `message` (required): The user's question/message
- `thread_id` (optional): Unique identifier for conversation continuity. Defaults to "default" if not provided. Use different thread_ids for different users to maintain separate conversation histories.

**Response:**
```json
{
  "response": "Sri Lanka offers stunning beaches...",
  "thread_id": "user_123",
  "status": "success"
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "status": "error"
}
```

### 3. Reset Conversation
**Endpoint:** `POST /reset`

**Request Body:**
```json
{
  "thread_id": "user_123"
}
```

**Response:**
```json
{
  "message": "Conversation reset requested for thread: user_123",
  "status": "success",
  "note": "Memory will be fresh with new thread_id"
}
```

## Node.js Integration Examples

### Using Axios

```javascript
const axios = require('axios');

// Send a chat message
async function sendMessage(message, userId) {
  try {
    const response = await axios.post('http://localhost:5000/chat', {
      message: message,
      thread_id: userId
    });
    
    return response.data.response;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Usage
sendMessage("Tell me about Colombo", "user_123")
  .then(reply => console.log(reply))
  .catch(err => console.error(err));
```

### Using Fetch (Native)

```javascript
async function sendMessage(message, userId) {
  try {
    const response = await fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        thread_id: userId
      })
    });
    
    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

### Express.js Backend Integration

```javascript
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const CHATBOT_API = 'http://localhost:5000';

// Endpoint for your frontend
app.post('/api/chat', async (req, res) => {
  try {
    const { message, userId } = req.body;
    
    const response = await axios.post(`${CHATBOT_API}/chat`, {
      message: message,
      thread_id: userId || 'default'
    });
    
    res.json({
      success: true,
      reply: response.data.response
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(3000, () => {
  console.log('Node.js backend running on port 3000');
});
```

## Important Notes

1. **Thread IDs**: Use unique thread_ids for each user to maintain separate conversation histories. The chatbot remembers previous messages within the same thread.

2. **CORS**: CORS is enabled by default. The Flask app accepts requests from any origin.

3. **Memory**: The chatbot uses MemorySaver which stores conversation history in memory. In production, consider using MongoDB checkpointer for persistent storage.

4. **API Keys**: Make sure all API keys in the Flask app are valid (OpenAI, Tavily, MongoDB).

5. **Performance**: First request may be slower due to model loading. Subsequent requests will be faster.

## Testing with cURL

```bash
# Health check
curl http://localhost:5000/health

# Send a message
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Sigiriya", "thread_id": "test_user"}'
```

## Production Considerations

1. **Environment Variables**: Move API keys to environment variables or .env file
2. **Authentication**: Add authentication middleware to protect endpoints
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Persistent Storage**: Use MongoDB checkpointer instead of MemorySaver
5. **Load Balancing**: Deploy multiple instances behind a load balancer
6. **Logging**: Add comprehensive logging for debugging and monitoring
