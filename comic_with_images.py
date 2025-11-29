# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image
# import io

# # Cache models
# @st.cache_resource
# def load_text_model():
#     model_name = "gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer, model

# @st.cache_resource
# def load_image_model():
#     # Using a smaller, faster SD model for local generation
#     model_id = "CompVis/stable-diffusion-v1-4"
    
#     # Check if CUDA is available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#         safety_checker=None  # Disable for speed
#     )
#     pipe = pipe.to(device)
    
#     # Optimize for speed
#     if device == "cpu":
#         pipe.enable_attention_slicing()
    
#     return pipe

# def generate_comic_script(idea, genre, tone, tokenizer, model):
#     """Generate 4-panel comic script with better control"""
    
#     # Try to generate continuation for each panel separately for better control
#     panels = []
    
#     # Panel 1 - Setup (use the idea directly)
#     panel1 = f"We see {idea.lower()}"
#     panels.append(panel1)
    
#     # Panel 2 - Generate conflict
#     prompt2 = f"""Continue this comic story:
# Panel 1: {panel1}
# Panel 2: Something goes wrong -"""
    
#     inputs = tokenizer(prompt2, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=len(inputs.input_ids[0]) + 30,
#             temperature=0.7,
#             top_k=40,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#             num_return_sequences=1
#         )
#     panel2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     panel2 = panel2.split("Panel 2:")[-1].split("Panel")[0].strip()[:100]
#     panels.append(panel2 if panel2 else "Things get complicated")
    
#     # Panel 3 - Generate twist
#     prompt3 = f"""Continue this comic:
# Panel 1: {panel1}
# Panel 2: {panels[1]}
# Panel 3: Unexpected twist -"""
    
#     inputs = tokenizer(prompt3, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=len(inputs.input_ids[0]) + 30,
#             temperature=0.8,
#             top_k=40,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#             num_return_sequences=1
#         )
#     panel3 = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     panel3 = panel3.split("Panel 3:")[-1].split("Panel")[0].strip()[:100]
#     panels.append(panel3 if panel3 else "A surprising turn of events")
    
#     # Panel 4 - Generate ending
#     prompt4 = f"""Continue this comic:
# Panel 1: {panel1}
# Panel 2: {panels[1]}
# Panel 3: {panels[2]}
# Panel 4: {tone} ending -"""
    
#     inputs = tokenizer(prompt4, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=len(inputs.input_ids[0]) + 30,
#             temperature=0.8,
#             top_k=50,
#             top_p=0.95,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#             num_return_sequences=1
#         )
#     panel4 = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     panel4 = panel4.split("Panel 4:")[-1].split("Panel")[0].strip()[:100]
#     panels.append(panel4 if panel4 else "Everything works out")
    
#     # Return formatted script
#     return "\n".join([f"Panel {i+1}: {p}" for i, p in enumerate(panels)])

# def parse_script(raw_script):
#     """Parse script into structured panels"""
#     panels = []
    
#     # Split by Panel markers
#     lines = raw_script.split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if line.startswith('Panel') and ':' in line:
#             # Extract text after "Panel X:"
#             text = line.split(':', 1)[1].strip()
#             if text and not text.startswith('http'):
#                 # Clean and limit length
#                 text = text[:150].strip()
#                 panels.append(text)
    
#     # Ensure we have exactly 4 panels
#     while len(panels) < 4:
#         panels.append("A scene from the story")
    
#     return panels[:4]

# def create_image_prompt(scene_text, art_style):
#     """Convert scene description to image generation prompt"""
#     # Clean the text - take only first sentence or first 100 chars
#     import re
    
#     # Remove URLs
#     clean_text = re.sub(r'http\S+', '', scene_text)
    
#     # Take only first sentence (up to period, question mark, or exclamation)
#     sentences = re.split(r'[.!?]', clean_text)
#     if sentences:
#         core_description = sentences[0].strip()
#     else:
#         core_description = clean_text[:100].strip()
    
#     # Limit to reasonable length (Stable Diffusion works best with short prompts)
#     if len(core_description) > 100:
#         core_description = core_description[:100].strip()
    
#     # Build focused prompt
#     enhanced_prompt = f"{art_style}, {core_description}, single panel comic art, clear focus, professional illustration, vibrant colors"
    
#     return enhanced_prompt

# def generate_panel_image(prompt, pipe, art_style):
#     """Generate image for a comic panel"""
#     try:
#         # Add negative prompt for better quality
#         negative_prompt = "blurry, low quality, distorted, ugly, text, watermark"
        
#         image = pipe(
#             prompt,
#             negative_prompt=negative_prompt,
#             num_inference_steps=20,  # Lower for speed (increase to 50 for quality)
#             guidance_scale=7.5,
#             height=512,
#             width=512
#         ).images[0]
        
#         return image
#     except Exception as e:
#         st.error(f"Image generation error: {str(e)}")
#         # Return a placeholder image
#         return Image.new('RGB', (512, 512), color='lightgray')

# # Streamlit UI
# st.set_page_config(page_title="🎬 Mini Movie Generator", page_icon="🎬", layout="wide")

# st.title("🎬 Mini Movie Generator")
# st.markdown("*Transform any idea into a 4-panel comic with AI-generated images!*")

# # Sidebar settings
# st.sidebar.header("⚙️ Settings")

# genre = st.sidebar.selectbox(
#     "Genre",
#     ["comedy", "horror-comedy", "romance", "sci-fi", "slice-of-life", "adventure"]
# )

# tone = st.sidebar.selectbox(
#     "Tone",
#     ["funny", "wholesome", "absurd", "dramatic", "sarcastic"]
# )

# art_style = st.sidebar.selectbox(
#     "Art Style",
#     [
#         "comic book style",
#         "cartoon style", 
#         "anime style",
#         "pixel art style",
#         "watercolor style",
#         "sketch style"
#     ]
# )

# enable_images = st.sidebar.checkbox("🎨 Generate Images", value=True, 
#                                     help="Uncheck for faster text-only generation")

# # Load models
# st.sidebar.markdown("---")
# st.sidebar.subheader("🤖 Model Status")

# with st.spinner("Loading text model..."):
#     tokenizer, text_model = load_text_model()
#     st.sidebar.success("✅ Text model loaded")

# if enable_images:
#     with st.spinner("Loading image model... (this may take a minute)"):
#         try:
#             image_pipe = load_image_model()
#             st.sidebar.success("✅ Image model loaded")
#             device_type = "GPU" if torch.cuda.is_available() else "CPU"
#             st.sidebar.info(f"Running on: {device_type}")
#         except Exception as e:
#             st.sidebar.error("❌ Image model failed to load")
#             st.sidebar.warning("Continuing with text-only mode")
#             enable_images = False

# # Example ideas
# st.sidebar.markdown("---")
# st.sidebar.subheader("💡 Example Ideas")
# example_ideas = [
#     "Two friends fighting over who ate the last biryani",
#     "A cat discovering it's actually a dragon",
#     "Programmer debugging code at 3 AM",
#     "Student trying to stay awake in boring lecture",
#     "Dog trying to catch its own tail"
# ]

# if st.sidebar.button("🎲 Random Idea"):
#     import random
#     st.session_state.random_idea = random.choice(example_ideas)

# # Main input
# col_input1, col_input2 = st.columns([3, 1])

# with col_input1:
#     idea = st.text_area(
#         "Enter your comic idea:",
#         value=st.session_state.get('random_idea', ''),
#         placeholder="e.g., A superhero who's afraid of heights",
#         height=100
#     )

# with col_input2:
#     st.write("")
#     st.write("")
#     generate_btn = st.button("🎬 Generate Comic", type="primary", use_container_width=True)

# if generate_btn and idea:
    
#     # Step 1: Generate script
#     with st.spinner("📝 Writing your story..."):
#         raw_script = generate_comic_script(idea, genre, tone, tokenizer, text_model)
#         panels = parse_script(raw_script)
    
#     st.success("✅ Script generated!")
    
#     # Display the comic
#     st.markdown("---")
#     st.subheader("🎬 Your 4-Panel Comic")
    
#     panel_titles = ["Setup", "Conflict", "Twist", "Punchline"]
#     panel_emojis = ["🎬", "⚡", "🎭", "😂"]
    
#     # Create 2x2 grid for panels
#     row1_col1, row1_col2 = st.columns(2)
#     row2_col1, row2_col2 = st.columns(2)
    
#     columns = [row1_col1, row1_col2, row2_col1, row2_col2]
    
#     for i, (col, panel_text) in enumerate(zip(columns, panels)):
#         with col:
#             st.markdown(f"### {panel_emojis[i]} Panel {i+1}: {panel_titles[i]}")
#             st.write(panel_text)
            
#             if enable_images:
#                 with st.spinner(f"🎨 Drawing panel {i+1}..."):
#                     # Create image prompt
#                     img_prompt = create_image_prompt(panel_text, art_style)
                    
#                     # Generate image
#                     image = generate_panel_image(img_prompt, image_pipe, art_style)
                    
#                     # Display image
#                     st.image(image, use_container_width=True)
                    
#                     # Show prompt used (in expander)
#                     with st.expander("🔍 Image prompt"):
#                         st.caption(img_prompt)
            
#             st.markdown("---")
    
#     # Show raw script
#     with st.expander("📜 View raw script"):
#         st.text(raw_script)

# elif generate_btn:
#     st.warning("⚠️ Please enter an idea first!")

# # Tips section
# st.markdown("---")
# with st.expander("💡 Tips for Best Results"):
#     st.markdown("""
#     **For Better Stories:**
#     - Be specific: "Two roommates fight over AC temperature" > "Friends argue"
#     - Add character details: "A clumsy wizard tries cooking"
#     - Include a setting: "In a busy coffee shop, someone spills coffee on their laptop"
    
#     **For Better Images:**
#     - Simpler scenes work better: "Person sitting at desk" > "Complex crowd scene"
#     - Avoid too many characters in one panel (2-3 max)
#     - Use clear art styles: cartoon, anime, comic book
    
#     **Performance:**
#     - First generation takes longer (models loading)
#     - CPU mode: ~30-60 seconds per image
#     - GPU mode: ~5-10 seconds per image
#     - Disable images for instant script generation
#     """)

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style="text-align: center; color: #666; font-size: 14px;">
#         <p>🎨 Powered by GPT-2 (text) + Stable Diffusion (images)</p>
#         <p>Built for NLP Project Demo</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Cache models
@st.cache_resource
def load_text_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

@st.cache_resource
def load_image_model():
    # Using a smaller, faster SD model for local generation
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable for speed
    )
    pipe = pipe.to(device)
    
    # Optimize for speed
    if device == "cpu":
        pipe.enable_attention_slicing()
    
    return pipe

def generate_comic_script(idea, genre, tone, tokenizer, model):
    """Generate 4-panel comic script using single coherent prompt"""
    
    # Generate ALL panels in one go for better coherence
    prompt = f"""Write a very short {tone} {genre} comic story in exactly 4 panels about: {idea}

Each panel is ONE simple sentence describing what happens.

Panel 1 (Setup): {idea} in a normal situation
Panel 2 (Conflict): {idea} faces a problem
Panel 3 (Twist): Something unexpected happens
Panel 4 (Punchline): A {tone} resolution

Story:
Panel 1:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=250,
            temperature=0.75,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_script = generated_text[len(prompt):]
    
    # If generation fails or is too short, use template fallback
    if len(generated_script) < 50:
        generated_script = generate_template_fallback(idea, genre, tone)
    
    return generated_script

def generate_template_fallback(idea, genre, tone):
    """Fallback to template-based generation if GPT-2 fails"""
    import random
    
    # Extract main subject from idea
    subject = idea.lower().strip()
    
    # Template patterns for different genres
    templates = {
        'comedy': [
            f"Panel 1: {subject} going about their day normally\nPanel 2: {subject} encounters an embarrassing situation\nPanel 3: {subject} tries to fix it but makes it worse\nPanel 4: {subject} accepts the chaos and laughs it off",
            f"Panel 1: {subject} is feeling confident\nPanel 2: {subject} attempts something ambitious\nPanel 3: {subject} fails spectacularly\nPanel 4: {subject} realizes they're better off this way"
        ],
        'adventure': [
            f"Panel 1: {subject} discovers a mysterious object\nPanel 2: {subject} activates it accidentally\nPanel 3: {subject} is transported to a strange place\nPanel 4: {subject} finds a way back home",
            f"Panel 1: {subject} receives a quest\nPanel 2: {subject} faces a dangerous obstacle\nPanel 3: {subject} uses clever thinking to overcome it\nPanel 4: {subject} claims their reward"
        ],
        'slice-of-life': [
            f"Panel 1: {subject} starts a normal day\nPanel 2: {subject} notices something unusual\nPanel 3: {subject} investigates and learns something new\nPanel 4: {subject} appreciates the small moments",
            f"Panel 1: {subject} has a routine task to do\nPanel 2: {subject} gets distracted by something interesting\nPanel 3: {subject} forgets what they were doing\nPanel 4: {subject} doesn't mind at all"
        ],
        'sci-fi': [
            f"Panel 1: {subject} in a futuristic setting\nPanel 2: {subject} encounters a technology malfunction\nPanel 3: {subject} discovers it's not a bug but a feature\nPanel 4: {subject} adapts to the new reality",
            f"Panel 1: {subject} living in a high-tech world\nPanel 2: {subject} questions the system\nPanel 3: {subject} finds a hidden truth\nPanel 4: {subject} decides what to do next"
        ],
        'romance': [
            f"Panel 1: {subject} sees someone interesting\nPanel 2: {subject} tries to get their attention\nPanel 3: {subject} makes an awkward first impression\nPanel 4: They both laugh and connect anyway",
            f"Panel 1: {subject} on a date\nPanel 2: {subject} tries too hard to impress\nPanel 3: {subject} relaxes and acts naturally\nPanel 4: The date goes wonderfully"
        ]
    }
    
    # Get templates for the genre, default to comedy
    genre_templates = templates.get(genre, templates['comedy'])
    
    # Pick random template
    return random.choice(genre_templates)

def parse_script(raw_script):
    """Parse script into structured panels"""
    panels = []
    
    # Split by Panel markers or newlines
    lines = raw_script.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Panel') and ':' in line:
            # Extract text after "Panel X:"
            text = line.split(':', 1)[1].strip()
            if text and not text.startswith('http') and len(text) > 3:
                # Clean and limit length
                text = text[:200].strip()
                panels.append(text)
    
    # If no panels found, try splitting by newlines
    if len(panels) == 0:
        lines = [l.strip() for l in raw_script.split('\n') if l.strip() and not l.startswith('Panel')]
        panels = lines[:4]
    
    # Ensure we have exactly 4 panels
    while len(panels) < 4:
        panels.append("Continuing the story...")
    
    return panels[:4]

def create_image_prompt(scene_text, art_style):
    """Convert scene description to image generation prompt"""
    # Clean the text - take only first sentence or first 100 chars
    import re
    
    # Remove URLs
    clean_text = re.sub(r'http\S+', '', scene_text)
    
    # Take only first sentence (up to period, question mark, or exclamation)
    sentences = re.split(r'[.!?]', clean_text)
    if sentences:
        core_description = sentences[0].strip()
    else:
        core_description = clean_text[:100].strip()
    
    # Limit to reasonable length (Stable Diffusion works best with short prompts)
    if len(core_description) > 100:
        core_description = core_description[:100].strip()
    
    # Build focused prompt
    enhanced_prompt = f"{art_style}, {core_description}, single panel comic art, clear focus, professional illustration, vibrant colors"
    
    return enhanced_prompt

def generate_panel_image(prompt, pipe, art_style):
    """Generate image for a comic panel"""
    try:
        # Add negative prompt for better quality
        negative_prompt = "blurry, low quality, distorted, ugly, text, watermark"
        
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,  # Lower for speed (increase to 50 for quality)
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        return image
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        # Return a placeholder image
        return Image.new('RGB', (512, 512), color='lightgray')

# Streamlit UI
st.set_page_config(page_title="🎬 Mini Movie Generator", page_icon="🎬", layout="wide")

st.title("🎬 Mini Movie Generator")
st.markdown("*Transform any idea into a 4-panel comic with AI-generated images!*")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

story_mode = st.sidebar.radio(
    "Story Generation",
    ["AI Generation", "Template-Based (More Coherent)"],
    help="Template mode guarantees story coherence"
)

genre = st.sidebar.selectbox(
    "Genre",
    ["comedy", "horror-comedy", "romance", "sci-fi", "slice-of-life", "adventure"]
)

tone = st.sidebar.selectbox(
    "Tone",
    ["funny", "wholesome", "absurd", "dramatic", "sarcastic"]
)

art_style = st.sidebar.selectbox(
    "Art Style",
    [
        "comic book style",
        "cartoon style", 
        "anime style",
        "pixel art style",
        "watercolor style",
        "sketch style"
    ]
)

enable_images = st.sidebar.checkbox("🎨 Generate Images", value=True, 
                                    help="Uncheck for faster text-only generation")

# Load models
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Status")

with st.spinner("Loading text model..."):
    tokenizer, text_model = load_text_model()
    st.sidebar.success("✅ Text model loaded")

if enable_images:
    with st.spinner("Loading image model... (this may take a minute)"):
        try:
            image_pipe = load_image_model()
            st.sidebar.success("✅ Image model loaded")
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            st.sidebar.info(f"Running on: {device_type}")
        except Exception as e:
            st.sidebar.error("❌ Image model failed to load")
            st.sidebar.warning("Continuing with text-only mode")
            enable_images = False

# Example ideas
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example Ideas")
example_ideas = [
    "Two friends fighting over who ate the last biryani",
    "A cat discovering it's actually a dragon",
    "Programmer debugging code at 3 AM",
    "Student trying to stay awake in boring lecture",
    "Dog trying to catch its own tail"
]

if st.sidebar.button("🎲 Random Idea"):
    import random
    st.session_state.random_idea = random.choice(example_ideas)

# Main input
col_input1, col_input2 = st.columns([3, 1])

with col_input1:
    idea = st.text_area(
        "Enter your comic idea:",
        value=st.session_state.get('random_idea', ''),
        placeholder="e.g., A superhero who's afraid of heights",
        height=100
    )

with col_input2:
    st.write("")
    st.write("")
    generate_btn = st.button("🎬 Generate Comic", type="primary", use_container_width=True)

if generate_btn and idea:
    
    # Step 1: Generate script
    with st.spinner("📝 Writing your story..."):
        if story_mode == "Template-Based (More Coherent)":
            # Use template for guaranteed coherence
            raw_script = generate_template_fallback(idea, genre, tone)
        else:
            # Try AI generation
            raw_script = generate_comic_script(idea, genre, tone, tokenizer, text_model)
        
        panels = parse_script(raw_script)
    
    st.success("✅ Script generated!")
    
    # Display the comic
    st.markdown("---")
    st.subheader("🎬 Your 4-Panel Comic")
    
    panel_titles = ["Setup", "Conflict", "Twist", "Punchline"]
    panel_emojis = ["🎬", "⚡", "🎭", "😂"]
    
    # Create 2x2 grid for panels
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    columns = [row1_col1, row1_col2, row2_col1, row2_col2]
    
    for i, (col, panel_text) in enumerate(zip(columns, panels)):
        with col:
            st.markdown(f"### {panel_emojis[i]} Panel {i+1}: {panel_titles[i]}")
            st.write(panel_text)
            
            if enable_images:
                with st.spinner(f"🎨 Drawing panel {i+1}..."):
                    # Create image prompt
                    img_prompt = create_image_prompt(panel_text, art_style)
                    
                    # Generate image
                    image = generate_panel_image(img_prompt, image_pipe, art_style)
                    
                    # Display image
                    st.image(image, use_container_width=True)
                    
                    # Show prompt used (in expander)
                    with st.expander("🔍 Image prompt"):
                        st.caption(img_prompt)
            
            st.markdown("---")
    
    # Show raw script
    with st.expander("📜 View raw script"):
        st.text(raw_script)

elif generate_btn:
    st.warning("⚠️ Please enter an idea first!")

# Tips section
st.markdown("---")
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    **For Better Stories:**
    - Be specific: "Two roommates fight over AC temperature" > "Friends argue"
    - Add character details: "A clumsy wizard tries cooking"
    - Include a setting: "In a busy coffee shop, someone spills coffee on their laptop"
    
    **For Better Images:**
    - Simpler scenes work better: "Person sitting at desk" > "Complex crowd scene"
    - Avoid too many characters in one panel (2-3 max)
    - Use clear art styles: cartoon, anime, comic book
    
    **Performance:**
    - First generation takes longer (models loading)
    - CPU mode: ~30-60 seconds per image
    - GPU mode: ~5-10 seconds per image
    - Disable images for instant script generation
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>🎨 Powered by GPT-2 (text) + Stable Diffusion (images)</p>
        <p>Built for NLP Project Demo</p>
    </div>
    """,
    unsafe_allow_html=True
)