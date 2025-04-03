from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_pymongo import PyMongo
import google.generativeai as genai
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from flask_bcrypt import Bcrypt
from bson.objectid import ObjectId
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableLambda

import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
CORS(app)  # Allow frontend to access backend
bcrypt = Bcrypt(app)
mongo = PyMongo(app)

users_collection = mongo.db.users
chats_collection = mongo.db.chats

app.permanent_session_lifetime = timedelta(days=1)

# Configure Gemini AI API
my_api_key_gemini = os.getenv("GEMINI_API_KEY")
if not my_api_key_gemini:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=my_api_key_gemini)

try:
    model = genai.GenerativeModel("gemini-pro")
    logger.info("Gemini model initialized successfully.")
except Exception as model_error:
    logger.error(f"Model Initialization Error: {model_error}")
    raise RuntimeError("Failed to initialize Gemini AI model.")

# Custom Output Parser for Gemini
class GeminiOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text.strip()

# Wrap Gemini model in a RunnableLambda
def gemini_generate(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        raise

gemini_runnable = RunnableLambda(gemini_generate)

# LangChain Integration
template = """
You are a Agriculture expert. Provide detailed and accurate information about the agriculture :

{question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(llm=gemini_runnable, prompt=prompt, output_parser=GeminiOutputParser())

@app.route("/")
def home():
    return render_template('login.html')

# User Registration
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    user_id = users_collection.insert_one({"email": email, "password": hashed_password}).inserted_id
    
    return jsonify({"message": "User registered successfully", "user_id": str(user_id)})

# User Login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    user = users_collection.find_one({"email": email})
    if user and bcrypt.check_password_hash(user["password"], password):
        session["user_id"] = str(user["_id"])
        session["email"] = email  # Store email in session
        return jsonify({"message": "Login successful", "redirect": "/chat"}), 200
    
    return jsonify({"error": "Invalid email or password"}), 401

# User Logout
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    session.pop("email", None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route("/chat")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))  # Redirect to login if session is missing
    return render_template("index.html", email=session["email"])

SOLAR_SYSTEM_KEYWORDS = ["Crop Yield","hii" ," hello"
"Precision Farming","soil","water","soil nutreints","Agricultural Forecasting","Fertilizer Recommendation","Pest Management","Irrigation Optimization""Soil Health Monitoring","Weather Prediction","Sustainable Farming","Remote Sensing in Agriculture"]

def is_solar_system_related(prompt):
    return any(keyword in prompt.lower() for keyword in SOLAR_SYSTEM_KEYWORDS)

@app.route("/ask", methods=["POST"])
def ask():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        prompt_text = request.json.get("prompt")
        if not prompt_text:
            return jsonify({"error": "No prompt provided"}), 400

        if not is_solar_system_related(prompt_text):
            return jsonify({"error": "I'm programmed to answer only agriculture questions."}), 400

        try:
            # Log the prompt being sent to Gemini
            logger.debug(f"Sending prompt to Gemini API: {prompt_text}")
            
            # Generate response using Gemini API
            response = model.generate_content(prompt_text)
            
            # Log the response from Gemini
            logger.debug(f"Response from Gemini API: {response.text}")
            
            answer = response.text.strip() if response.text else "I'm sorry, I couldn't generate a response."
            formatted_answer = answer  # Gemini already generates Markdown
        except Exception as api_error:
            logger.error(f"Gemini API Error: {api_error}")
            return jsonify({"error": "Failed to get a response from Gemini API. Please try again later."}), 500

        try:
            # Save the chat to MongoDB
            chat = {
                "user_id": session["user_id"],
                "question": prompt_text,
                "answer": formatted_answer,
                "likes": 0,
                "dislikes": 0
            }
            chat_id = chats_collection.insert_one(chat).inserted_id
            return jsonify({"data": formatted_answer, "chat_id": str(chat_id)}), 200
        except Exception as mongo_error:
            logger.error(f"MongoDB Error: {mongo_error}")
            return jsonify({"error": "Failed to save chat to database."}), 500
    except Exception as e:
        logger.error(f"General Error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Edit User's Query
@app.route("/edit_query/<chat_id>", methods=["PUT"])
def edit_query(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        data = request.json
        new_query = data.get("new_query")
        if not new_query:
            return jsonify({"error": "No new query provided"}), 400

        chats_collection.update_one({"_id": ObjectId(chat_id)}, {"$set": {"question": new_query}})
        return jsonify({"message": "Query updated successfully"}), 200
    except Exception as e:
        logger.error(f"Error updating query: {e}")
        return jsonify({"error": "Failed to update query."}), 500

# Delete Individual Chat
@app.route("/delete_chat/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        chats_collection.delete_one({"_id": ObjectId(chat_id)})
        return jsonify({"message": "Chat deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        return jsonify({"error": "Failed to delete chat."}), 500

# Like/Dislike Chat Response
@app.route("/rate_chat/<chat_id>", methods=["POST"])
def rate_chat(chat_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        data = request.json
        action = data.get("action")  # "like" or "dislike"

        Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        update_field = "likes" if action == "like" else "dislikes"
        chats_collection.update_one({"_id": ObjectId(chat_id)}, {"$inc": {update_field: 1}})
        return jsonify({"message": f"Chat {action}d successfully"}), 200
    except Exception as e:
        logger.error(f"Error rating chat: {e}")
        return jsonify({"error": "Failed to rate chat."}), 500

# Fetch Chat History for Logged-in User
@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        chats = list(chats_collection.find({"user_id": session["user_id"]}, {"_id": 1, "question": 1, "answer": 1, "likes": 1, "dislikes": 1}))
        return jsonify({"chat_history": chats}), 200
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return jsonify({"error": "Failed to fetch chat history."}), 500

# Delete All Chat History
@app.route("/delete_chat_history", methods=["DELETE"])
def delete_chat_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    try:
        chats_collection.delete_many({"user_id": session["user_id"]})
        return jsonify({"message": "Chat history deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        return jsonify({"error": "Failed to delete chat history."}), 500

if __name__ == "__main__":
    app.run(debug=False)