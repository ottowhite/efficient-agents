import os
import asyncio
from src.utils import monitor_event_loop_lag
from quart import Quart, request, jsonify
from dotenv import load_dotenv
from src.utils import custom_chat_template
from src.examples.beam_search.models import LLM, PRM, Tokenizer
from src.examples.beam_search.searches import BeamSearch

# Initialize Flask app
app = Quart(__name__)

# Global variables for models (initialized on startup)
llm = None
prm = None

def initialize_models():
    """Initialize LLM and PRM models"""
    global llm, prm
    
    load_dotenv()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = Tokenizer(model_name, custom_chat_template)
    
    llm = LLM(
        model_name=model_name,
        base_url="http://llama1b-llm:8000/v1",
        temperature=0.8,
        tokenizer=tokenizer
    )

    prm = PRM(
        model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        base_url="http://llama8b-prm:8000/v1"
    )

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "models_initialized": llm is not None and prm is not None})

@app.post('/beam_search')
async def beam_search_endpoint():
    """
    Beam search endpoint that accepts a problem and returns beam search results
    
    Expected JSON payload:
    {
        "problem": "What is 2+2?",
        "search_width": 4,
        "select_top_k": 1,
        "max_iterations": 40
    }
    
    Returns:
    {
        "thoughts": [list of thought dictionaries with steps and scores]
    }
    """
    try:
        # Check if models are initialized
        if llm is None or prm is None:
            return jsonify({"error": "Models not initialized"}), 500
        
        # Parse request
        data = await request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        problem = data.get("problem")
        if not problem:
            return jsonify({"error": "No problem provided"}), 400
        
        # Get search parameters with defaults
        search_width = data.get("search_width", 4)
        select_top_k = data.get("select_top_k", 1)
        max_iterations = data.get("max_iterations", 40)

        beam_search = BeamSearch(
            problem=problem,
            llm=llm,
            prm=prm,
            search_width=search_width,
            select_top_k=select_top_k,
            max_iterations=max_iterations
        )

        thoughts = await beam_search.run()

        return jsonify([thought.to_dict() for thought in thoughts])
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.before_serving
async def startup_event():
    loop = asyncio.get_event_loop()
    asyncio.create_task(monitor_event_loop_lag(loop))

if __name__ == '__main__':
    print("Initializing models...")
    initialize_models()
    port = int(os.environ.get("PORT", "5000"))
    print(f"Models initialized. Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
