import streamlit as st
from collections import deque
import sys
import os

# Add the parent directory to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chat_utils import initialize_gemini_chat, get_gemini_response, extract_math_expression
from calculator import calculate_sympy  # Use the SymPy based calculator

MAX_HISTORY_LEN = 10  # Max user-bot exchanges in sidebar history

# --- Page Configuration ---
st.set_page_config(page_title="Gemini-SymPy Chatbot 🤖🧮", layout="wide", initial_sidebar_state="expanded")

# --- Title ---
st.title("Gemini Chatbot with SymPy Calculator 🤖🧮")
st.caption("Ask general questions or perform calculations (e.g., `sqrt(16) + sin(pi/2)`).")

# --- Session State Initialization ---
if "messages" not in st.session_state:  # For main chat display {role: "user/assistant", content: "...">
    st.session_state.messages = []
if "history" not in st.session_state:  # For sidebar display (user_query, bot_response)
    st.session_state.history = deque(maxlen=MAX_HISTORY_LEN)
if "llm_answers" not in st.session_state:  # Store Gemini responses for math queries
    st.session_state.llm_answers = {}
if "last_math_query" not in st.session_state:  # Track last math query for LLM answer
    st.session_state.last_math_query = None
if "rerun_flag" not in st.session_state:  # Flag to control rerun behavior
    st.session_state.rerun_flag = False

# Initialize Gemini chat session once
if "gemini_chat_session" not in st.session_state:
    model, chat = initialize_gemini_chat()
    if model and chat:
        st.session_state.gemini_model = model
        st.session_state.gemini_chat_session = chat
    else:
        st.error("🔴 Failed to initialize Gemini. Please check your API key in .env and console logs.")
        st.session_state.gemini_model = None
        st.session_state.gemini_chat_session = None

# --- Sidebar for Message History ---
with st.sidebar:
    # New Chat button at the top
    if st.button("🆕 New Chat", use_container_width=True, help="Start a new conversation"):
        st.session_state.messages = []
        st.session_state.history = deque(maxlen=MAX_HISTORY_LEN)
        st.session_state.llm_answers = {}
        st.session_state.last_math_query = None
        # Reset Gemini chat session
        model, chat = initialize_gemini_chat()
        if model and chat:
            st.session_state.gemini_model = model
            st.session_state.gemini_chat_session = chat
        st.rerun()
    
    st.divider()
    
    st.header("📜 Conversation History")
    st.caption(f"Last {MAX_HISTORY_LEN} exchanges (newest first):")
    if not st.session_state.history:
        st.info("No conversation history yet.")
    else:
        # Display newest first by iterating over a reversed copy
        for i, (user_q, bot_r) in enumerate(reversed(list(st.session_state.history))):
            with st.expander(f"Exchange {len(st.session_state.history) - i}", expanded=False):
                st.markdown(f"**You:** {user_q}")
                st.markdown(f"**Bot:** {bot_r}")
            st.divider()

# --- Main Chat Area ---
# Display existing messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)  # Allow HTML for icons

# Handle new user input
if prompt := st.chat_input("Your message or calculation..."):
    st.session_state.rerun_flag = False  # Reset rerun flag

    # Add user message to main chat display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    user_message_for_history = prompt  # Preserve original prompt for history
    bot_response_content = ""
    is_local_calculation_success = False

    # Reset last math query
    st.session_state.last_math_query = None

    # Attempt to identify and process a math query using LLM/regex extraction
    if st.session_state.gemini_chat_session is None:
        error_message = "🔴 Gemini is not available. Cannot process queries."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.session_state.history.append((user_message_for_history, error_message))
        st.session_state.rerun_flag = True
    else:
        math_expression_candidate = extract_math_expression(prompt, st.session_state.gemini_chat_session)

        if math_expression_candidate:
            with st.spinner("🧮 Calculating..."):
                calculated_result = calculate_sympy(math_expression_candidate)
            
            if not calculated_result.startswith("Error:"):
                bot_response_content = f"🧮 **Result:** `{math_expression_candidate}` = **{calculated_result}**"
                is_local_calculation_success = True
                
                # Store this as a math query for potential LLM answer
                st.session_state.last_math_query = {
                    "query": prompt,
                    "expression": math_expression_candidate,
                    "result": calculated_result
                }
            else:
                # If SymPy returns an error, show the error but don't prevent Gemini from answering
                bot_response_content = f"⚠️ **SymPy Error:** For input `{math_expression_candidate}`: {calculated_result}"
                is_local_calculation_success = False
        else:
            # Not a math problem, reply with a prompt
            bot_response_content = "Please input a mathematical problem for me to solve."
            st.session_state.messages.append({"role": "assistant", "content": bot_response_content})
            st.session_state.history.append((user_message_for_history, bot_response_content))
            st.session_state.rerun_flag = True

        # If not a math problem or failed calculation, forward to Gemini
        if not math_expression_candidate or not is_local_calculation_success:
            with st.spinner("🤖 ChatBot is thinking..."):
                gemini_response_text = get_gemini_response(st.session_state.gemini_chat_session, prompt)
                bot_response_content = gemini_response_text
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response_content})
            st.session_state.history.append((user_message_for_history, bot_response_content))
            st.session_state.rerun_flag = True

    if st.session_state.rerun_flag:
        st.rerun()

# Show LLM Answer button for math queries
if st.session_state.last_math_query and st.session_state.gemini_chat_session:
    query = st.session_state.last_math_query["query"]
    expression = st.session_state.last_math_query["expression"]
    result = st.session_state.last_math_query["result"]
    
    # Only show button if we haven't already shown the answer
    if query not in st.session_state.llm_answers:
        if st.button(
            f"🤖 Show LLM Explanation for: {expression[:50]}{'...' if len(expression) > 50 else ''}",
            key=f"llm_explain_{hash(expression)}",
            help="Get Gemini's explanation of this calculation"
        ):
            with st.spinner("🤖 Gemini is thinking about the math..."):
                # Ask Gemini specifically about the math expression
                math_prompt = (
                    f"Explain the calculation: {expression} = {result}. "
                    f"Provide a step-by-step explanation in simple terms."
                )
                gemini_response = get_gemini_response(
                    st.session_state.gemini_chat_session,
                    math_prompt
                )
                
                # Store the response
                st.session_state.llm_answers[query] = gemini_response
                st.experimental_rerun()

# Display LLM answers for math queries
for query, answer in st.session_state.llm_answers.items():
    with st.chat_message("assistant"):
        st.markdown(f"**🤖 Gemini Explanation for `{query[:50]}{'...' if len(query) > 50 else ''}`**")
        st.markdown(answer, unsafe_allow_html=True)

# Add a small footer or instruction
st.markdown("---")
st.markdown("Built with Streamlit, Gemini, and SymPy.")