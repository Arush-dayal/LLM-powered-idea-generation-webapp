import streamlit as st
from google import genai 
import os
from dotenv import load_dotenv
from idea_generation import (
    fetch_arxiv_papers,
    embed_papers,
    build_faiss_index,
    semantic_search,
    generate_ideas_with_gemini,
    generate_details_with_gemini
)

# Load API key from streamlit secrets or .env file
load_dotenv(r"secrets.env") 
gemini_api_key = os.getenv("GEMINI_API_KEY")


# --- Client Initialization ---
gemini_client_for_functions = genai.Client(api_key=gemini_api_key)

client_for_functions = gemini_client_for_functions 

st.set_page_config(page_title="AI Research Idea Generator", page_icon="🧠")
st.title("🧠 Smart AI Research Idea Generator")

st.markdown("Generate novel research questions grounded in recent arXiv papers.")

# Initialize session state variables
if 'ideas_generated' not in st.session_state:
    st.session_state.ideas_generated = False
    st.session_state.ideas_data = [] # To store the full idea objects
    st.session_state.idea_titles = []
    st.session_state.idea_descriptions = []
    st.session_state.selected_idea_details = None
    st.session_state.novelty_scores = []
    st.session_state.error_message = None


user_topic = st.text_input("🔍 Your area of research", "Graph neural networks for biology", key="user_topic_input")
keywords = st.text_input("🧷 Keywords or constraints", "protein interaction, interpretability", key="keywords_input")

if st.button("🚀 Generate Research Ideas"):

    # Reset session variables on new generation
    st.session_state.ideas_generated = False 
    st.session_state.ideas_data = []
    st.session_state.idea_titles = []
    st.session_state.idea_descriptions = []
    st.session_state.selected_idea_details = None
    st.session_state.novelty_scores = []
    st.session_state.error_message = None

    if not client_for_functions: # Check if the client (or configured module) is available
        st.error("❗ API key (e.g., GEMINI_API_KEY) is missing or client initialization failed. Please set it in secrets.env.")
        st.session_state.error_message = "API key missing or client initialization failed."
    elif not user_topic.strip():
        st.warning("Please enter an area of research.")
        st.session_state.error_message = "Area of research is empty."
    else:
        try:
            with st.spinner("🔎 Fetching recent arXiv papers..."):
                papers = fetch_arxiv_papers(user_topic + " " + keywords)
            
            if not papers:
                st.warning("No papers found for the given topic and keywords.")
                st.session_state.error_message = "No papers found."
            else:
                with st.spinner("📐 Embedding papers and building semantic index..."):
                    # Assuming embed_papers takes the genai module or a specific model
                    embeddings = embed_papers(client_for_functions, papers)
                    index = build_faiss_index(embeddings)
                    relevant_papers = semantic_search(client_for_functions, user_topic + " " + keywords, index, papers, top_k=5)

                with st.spinner("💡 Generating novel research ideas using Gemini..."):
                    # Assuming generate_ideas_with_gemini takes the genai module or a specific model
                    ideas = generate_ideas_with_gemini(user_topic, relevant_papers, client_for_functions)
                
                if ideas:
                    st.session_state.ideas_data = ideas # Store the raw idea objects
                    st.session_state.idea_titles = [i.title for i in ideas]
                    st.session_state.idea_descriptions = [i.brief_description for i in ideas]
                    st.session_state.ideas_generated = True
                else:
                    st.warning("Could not generate research ideas from the fetched papers.")
                    st.session_state.error_message = "Failed to generate ideas."

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.error_message = str(e)


# Display generated ideas if they exist in session state
if st.session_state.ideas_generated and st.session_state.idea_titles:
    st.markdown("### 💡 Suggested Research Ideas")
    # Displaying the ideas - assuming `ideas` were objects with title and brief_description
    for i, idea in enumerate(st.session_state.ideas_data):
        st.markdown(f"**{i+1}. {idea.title}**")
        st.markdown(idea.brief_description)
        st.markdown("---")

    # Form for choosing an idea and generating details
    with st.form(key="choose_idea_form"):
        selected_title_option = st.selectbox(
            label="✅ Choose a research idea to elaborate on:",
            options=st.session_state.idea_titles,
            index=0 # Default to the first option
        )
        generate_details_button = st.form_submit_button(label="🔬 Generate Further Details")
        
    if generate_details_button and selected_title_option:
        if not client_for_functions:
            st.error("❗ API key is missing or client initialization failed. Cannot generate details.")
        else:
            try:
                # Find the corresponding description
                idx = st.session_state.idea_titles.index(selected_title_option)
                selected_desc = st.session_state.idea_descriptions[idx]
                
                with st.spinner(f"🧬 Generating dataset and experiment design for: '{selected_title_option}'..."):
                    # Assuming generate_details_with_gemini takes title, description, and the client/model
                    output_details = generate_details_with_gemini(selected_title_option, selected_desc, client_for_functions)
                    st.session_state.selected_idea_details = output_details
            except Exception as e:
                st.error(f"An error occurred while generating details: {e}")
                st.session_state.selected_idea_details = None

elif st.session_state.error_message and not st.session_state.ideas_generated :
    pass 

if st.session_state.selected_idea_details:
    st.markdown("### 📊 Generated Details: Dataset and Experiment Design")
    st.markdown(st.session_state.selected_idea_details)

if not st.session_state.ideas_generated and st.session_state.error_message and not st.session_state.selected_idea_details:
     pass
