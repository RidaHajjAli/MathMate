import os
import re
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from config import GEMINI_MODEL_NAME, GEMINI_TEMPERATURE, GEMINI_TOP_P, GEMINI_TOP_K, GEMINI_MAX_TOKENS, GEMINI_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  
STANDALONE_MATH_REGEX = re.compile(
    r"""^
    (?=.*(?:\d|[πeE]|sqrt|log|ln|sin|cos|tan|cot|sec|csc|abs|factorial|gamma|zeta|beta|re|im|conjugate|arg|expand|factor|simplify|solve|diff|integrate|limit|sum|product|mean|stddev|variance|det|transpose|and|or|not|xor))
    [a-zA-Z0-9\s\.\+\-\*\/\(\)\^\%\,\_\=\<\>\!\&\|\{\}\[\]π∞]+
    $
    """, re.VERBOSE
)

def is_valid_math_expression(expr: str) -> bool:
    # Accepts only if contains at least one digit or known math symbol/function
    if not expr or len(expr) > 200:
        return False
    expr = expr.strip()
    if not STANDALONE_MATH_REGEX.fullmatch(expr):
        return False
    # Disallow sequences of only non-alphanum symbols (e.g., "____", "<<>>")
    if not re.search(r"[a-zA-Z0-9]", expr):
        return False
    # Disallow only underscores or brackets
    if re.fullmatch(r"[_\[\]\{\}\(\)\s]+", expr):
        return False
    return True

def extract_math_expression_llm(chat_session, query: str) -> str | None:
    """
    Uses Gemini LLM to extract a mathematical expression from a paragraph or sentence.
    Returns the extracted expression as a string, or None if not found or invalid.
    """
    if not chat_session:
        return None
    prompt = (
        "Given the following text, extract the main mathematical expression or equation to be calculated. "
        "Return only the expression (e.g., '5738*47'), nothing else. "
        "If there is no mathematical expression, return 'NONE'.\n\n"
        f"Text: {query}\n\nExpression:"
    )
    try:
        response = chat_session.send_message(prompt, timeout=GEMINI_TIMEOUT)
        expr = getattr(response, "text", None)
        if expr:
            expr = expr.strip().rstrip('.;:')
            if expr.upper() == "NONE" or expr == "":
                return None
            # Validate extracted expression
            if is_valid_math_expression(expr):
                return expr
            else:
                logger.warning(f"LLM returned invalid expression: {expr}")
                return None
        return None
    except Exception as e:
        logger.error(f"Error extracting math expression with LLM: {e}")
        return None

def extract_math_expression(query: str, chat_session=None) -> str | None:
    """
    Attempts to identify and extract a mathematical expression from the user's query.
    If not found by regex, uses Gemini LLM to extract from a paragraph.
    """
    query_clean = query.strip()
    if query_clean.endswith("?"):
        query_clean = query_clean[:-1].strip()
    query_lower = query_clean.lower()

    # 1. Try regex for standalone math expression
    if is_valid_math_expression(query_clean):
        return query_clean

    # 2. Try LLM extraction if regex fails
    if chat_session:
        expr = extract_math_expression_llm(chat_session, query)
        if expr:
            return expr
        return None

    return None

def initialize_gemini_chat():
    """
    Initializes and returns the Gemini API client and a new chat session.
    Returns (None, None) if API key is missing or configuration fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "top_p": GEMINI_TOP_P,
                "top_k": GEMINI_TOP_K,
                "max_output_tokens": GEMINI_MAX_TOKENS,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
        )
        chat_session = model.start_chat(history=[])
        return model, chat_session
    except Exception as e:
        logger.error(f"Error initializing Gemini: {e}")
        return None, None

def get_gemini_response(chat_session, user_prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns the response text.
    """
    if not chat_session:
        return "Error: Gemini chat session not initialized."
    try:
        response = chat_session.send_message(user_prompt)
        # Handle potential variations in response structure
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            logger.warning(f"Unexpected Gemini response structure: {response}")
            return "I received a response, but couldn't extract the text. Please check the logs."
    except Exception as e:
        logger.error(f"Error getting Gemini response: {e}")
        return f"⚠️ Error communicating with Gemini: {str(e)}"

if __name__ == '__main__':
    # Test math extraction
    math_tests = [
        "calculate 2 + 2",
        "what is sqrt(16) * (pi/2)",
        "solve 100 / (5-3)",
        "5 * (3+2)",
        "how much is 10% of 50",
        "what is the capital of France",
        "compute sin(pi/2) + cos(0)",
        "e^2",
        "log(100, 10)",
        "just a number 5",
        "what is 7",
        "calc 22/7",
        "find the derivative of x^2",
        "simplify (x+1)^2",
        "solve for x: 2x + 5 = 15"
    ]
    print("--- Math Extraction Tests ---")
    for mt in math_tests:
        expr = extract_math_expression(mt)
        print(f"Query: '{mt}' -> Extracted: '{expr}'")

    # Test Gemini (requires API key to be set in .env)
    print("\n--- Gemini Test (requires .env with API key) ---")
    _, test_chat = initialize_gemini_chat()
    if test_chat:
        print(f"User: Hello Gemini!")
        gemini_reply = get_gemini_response(test_chat, "Hello Gemini!")
        print(f"Gemini: {gemini_reply}")
    else:
        print("Skipping Gemini test as initialization failed (check API key).")