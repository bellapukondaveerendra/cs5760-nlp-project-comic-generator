import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cache the model to load only once
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Using base GPT-2 for speed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def generate_comic_script(idea, genre, tone, tokenizer, model):
    """Generate 4-panel comic script based on user input"""
    
    # Craft the prompt
    prompt = f"""Write a {tone} {genre} comic script in exactly 4 panels:

Idea: {idea}

Panel 1 (Setup): 
Panel 2 (Conflict): 
Panel 3 (Twist): 
Panel 4 (Punchline):

Panel 1 (Setup):"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=300,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    generated_script = generated_text[len(prompt):]
    
    return generated_script

def parse_script(raw_script):
    """Parse the generated script into panels"""
    panels = {
        "Panel 1 (Setup)": "",
        "Panel 2 (Conflict)": "",
        "Panel 3 (Twist)": "",
        "Panel 4 (Punchline)": ""
    }
    
    lines = raw_script.split('\n')
    current_panel = None
    
    for line in lines:
        line = line.strip()
        if "Panel 2" in line:
            current_panel = "Panel 2 (Conflict)"
        elif "Panel 3" in line:
            current_panel = "Panel 3 (Twist)"
        elif "Panel 4" in line:
            current_panel = "Panel 4 (Punchline)"
        elif current_panel and line:
            panels[current_panel] += line + " "
    
    return panels

# Streamlit UI
st.set_page_config(page_title="🎬 Mini Comic Generator", page_icon="🎬", layout="centered")

st.title("🎬 4-Panel Comic Script Generator")
st.markdown("*Transform any idea into a hilarious 4-panel comic script!*")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
genre = st.sidebar.selectbox(
    "Genre",
    ["comedy", "horror-comedy", "romance", "sci-fi", "slice-of-life", "thriller"]
)

tone = st.sidebar.selectbox(
    "Tone",
    ["funny", "wholesome", "absurd", "dark humor", "sarcastic", "dramatic"]
)

# Load model
with st.spinner("Loading AI model... (first time takes a minute)"):
    tokenizer, model = load_model()

st.success("✅ Model loaded! Ready to generate.")

# Example ideas
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example Ideas")
example_ideas = [
    "Two friends fighting over who ate the last biryani",
    "A student preparing for exam with zero motivation",
    "A cat trying to catch a laser pointer",
    "Someone trying to parallel park for the first time",
    "A programmer debugging code at 3 AM"
]

if st.sidebar.button("🎲 Random Idea"):
    import random
    st.session_state.random_idea = random.choice(example_ideas)

# Main input
idea = st.text_area(
    "Enter your comic idea:",
    value=st.session_state.get('random_idea', ''),
    placeholder="e.g., A student trying to wake up early for class",
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🎬 Generate Script", type="primary", use_container_width=True)

if generate_btn and idea:
    with st.spinner("🎨 Creating your comic script..."):
        try:
            # Generate script
            raw_script = generate_comic_script(idea, genre, tone, tokenizer, model)
            panels = parse_script(raw_script)
            
            # Display results
            st.markdown("---")
            st.subheader("📜 Your Comic Script")
            
            # Display each panel in a card-like format
            panel_emojis = ["🎬", "⚡", "🎭", "😂"]
            panel_colors = ["#FFE5E5", "#E5F3FF", "#FFF5E5", "#E5FFE5"]
            
            for i, (panel_name, content) in enumerate(panels.items()):
                if content.strip():  # Only show if content exists
                    st.markdown(
                        f"""
                        <div style="background-color: {panel_colors[i]}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4>{panel_emojis[i]} {panel_name}</h4>
                            <p style="font-size: 16px;">{content.strip()}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Show raw output for debugging (collapsible)
            with st.expander("🔍 View raw AI output"):
                st.text(raw_script)
                
        except Exception as e:
            st.error(f"Oops! Something went wrong: {str(e)}")
            st.info("Try rephrasing your idea or changing the settings.")

elif generate_btn:
    st.warning("⚠️ Please enter an idea first!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>💡 Tip: Try different genres and tones for varied results!</p>
        <p>Built with GPT-2 + Streamlit | NLP Project Demo</p>
    </div>
    """,
    unsafe_allow_html=True
)