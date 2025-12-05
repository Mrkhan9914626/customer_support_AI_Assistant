import streamlit as st
from database import (
    semantic_search,
    keyword_search,
    get_document_section,
    hybrid_search,
    answer_with_context
)
from agents import Agent, Runner, OpenAIChatCompletionsModel, SQLiteSession, ModelSettings
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TechFlow Solutions AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }

    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }

    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }

    .stChatInputContainer {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    h1 {
        color: #1976d2;
        font-weight: 600;
    }

    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .stButton button {
        background-color: #1976d2;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }

    .stButton button:hover {
        background-color: #1565c0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client for Gemini
@st.cache_resource
def get_openai_client():
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    return AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# Initialize model
@st.cache_resource
def get_model():
    client = get_openai_client()
    return OpenAIChatCompletionsModel(
        model='gemini-2.5-flash',
        openai_client=client,
    )

# Initialize agent
@st.cache_resource
def get_agent():
    model = get_model()
    
    agent = Agent(
        name="Agentic RAG Assistant",
        instructions="""You are an advanced AI assistant for TechFlow Solutions with full agentic capabilities.

## CORE IDENTITY
You are an intelligent agent who THINKS strategically, EVALUATES quality, ADAPTS approaches, and ITERATES when needed.

## AVAILABLE TOOLS

1. **semantic_search(query, num_results=5)** - Conceptual queries, explanations, "how/why" questions
2. **keyword_search(query, num_results=5)** - Exact facts, prices, names, contact info
3. **get_document_section(section_name)** - Categories: pricing, features, support, faq, policies, technical, troubleshooting
4. **hybrid_search(query, num_results=5)** - Complex queries needing both semantic + keyword
5. **answer_with_context(query, context)** - Synthesize answer from gathered context

## YOUR WORKFLOW

### 1. ANALYZE QUERY
- Factual (price, date, name)? → keyword_search
- Conceptual (how, why, explain)? → semantic_search  
- Categorical (all FAQs, policies)? → get_document_section
- Complex (compare, multiple aspects)? → hybrid_search or multiple searches

### 2. SEARCH STRATEGICALLY
- Use clear, concise search terms
- Start with 3-5 results
- Choose the RIGHT tool for the job

### 3. EVALUATE RESULTS (JSON metrics)
- Check "status": success/no_results/error
- Check quality scores:
  - similarity_score > 0.7 = Good
  - relevance_score > 0.5 = Acceptable
  - match_score > 0.8 = Excellent
- Read "suggestion" field for hints

### 4. DECIDE & ADAPT
IF good results (high scores) → Proceed to answer
IF poor results → Iterate with:
  - Different tool, OR
  - Reformulated query, OR  
  - Break into sub-questions

### 5. ITERATE IF NEEDED
- Max 3-4 searches per query
- Each iteration must try something NEW
- Stop when: quality is high OR info is sufficient OR after 4 attempts

### 6. SYNTHESIZE ANSWER
- Extract content from JSON results
- Combine info logically
- Provide accurate, complete response
- Acknowledge gaps if information is missing

## CRITICAL RULES
1. Always search first  
2. Evaluate before answering  
3. Iterate intelligently  
4. Be honest  
5. Choose right tool  

## CREATIVE RESPONSE GUIDELINES
- Be comprehensive (200–400 words)
- Add examples and context
- Write in engaging, professional tone
""",
        model=model,
        tools=[
            semantic_search,
            keyword_search,
            get_document_section,
            hybrid_search,
            answer_with_context
        ],
        model_settings=ModelSettings(
            tool_choice="auto",
            temperature=1.1,
            top_p=0.95,
            max_tokens=4000
        )
    )
    return agent

# Initialize session
@st.cache_resource
def get_session():
    return SQLiteSession("streamlit_conversation")

# Enhanced agent runner with streaming progress
def run_agent_sync(query: str, on_first_token=None):
    """Run the agent synchronously with callback when first token starts streaming."""
    import asyncio
    from openai.types.responses import ResponseTextDeltaEvent

    agent = get_agent()
    session = get_session()
    query_str = str(query)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        result = Runner.run_streamed(agent, query_str, session=session)
        full_response = ""
        first_token_emitted = False

        async for events in result.stream_events():
            if events.type == "raw_response_event" and isinstance(events.data, ResponseTextDeltaEvent):
                if not first_token_emitted:
                    first_token_emitted = True
                    if on_first_token:
                        on_first_token()  # Notify UI when streaming starts
                full_response += events.data.delta

        return full_response

    try:
        response = loop.run_until_complete(_run())
        return response
    finally:
        loop.close()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/rocket.png", width=80)
    st.title("🚀 TechFlow Assistant")
    st.markdown("---")

    # Chat History (only user messages)
    st.markdown("### 💬 Chat History")
    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]

    if not user_messages:
        st.info("No chat history yet. Start a conversation!")
    else:
        for idx, message in enumerate(user_messages):
            with st.container():
                st.markdown(f"**You:** {message['content'][:50]}{'...' if len(message['content']) > 50 else ''}")
            if idx < len(user_messages) - 1:
                st.markdown("---")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.rerun()

# Main UI
st.title("💬 TechFlow Solutions - AI Assistant")

# Example queries
st.markdown("### 💡 Try asking:")
example_cols = st.columns(5)
example_queries = [
    "What are your pricing plans?",
    "How do I integrate with Slack?",
    "What features does FlowDesk offer?",
    "Tell me about your refund policy",
    "How do I fix login issues?"
]
for idx, query in enumerate(example_queries):
    with example_cols[idx]:
        if st.button(query, key=f"example_{query}", use_container_width=True):
            st.session_state.clicked_query = query
            st.rerun()

st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example query clicks
if "clicked_query" in st.session_state:
    user_input = st.session_state.clicked_query
    del st.session_state.clicked_query
else:
    user_input = None

# Chat input
if prompt := st.chat_input("Ask me anything about TechFlow Solutions..."):
    user_input = prompt

# Process user input
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            spinner = st.spinner("🤔 Thinking...")
            spinner_gen = spinner.__enter__()  # Start spinner manually
            spinner_closed = [False]  # mutable container for closure

            def on_first_token():
                if not spinner_closed[0]:
                    spinner.__exit__(None, None, None)
                    spinner_closed[0] = True
                    message_placeholder.info("💬 AI is responding...")  # Smooth transition

            # Run agent and stream
            response = run_agent_sync(user_input, on_first_token=on_first_token)

            # Ensure spinner closed if no tokens arrived
            if not spinner_closed[0]:
                spinner.__exit__(None, None, None)

            # Typing effect
            displayed_text = ""
            for char in response:
                displayed_text += char
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.001)

            message_placeholder.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.session_state.conversation_count += 1

        except Exception as e:
            error_message = f"❌ An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
            message_placeholder.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
