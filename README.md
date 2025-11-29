🎬 Mini Movie Generator - Complete Setup Guide
==============================================

📋 Project Overview
-------------------

**Mini Movie Generator** creates 4-panel comic strips with both AI-generated story scripts AND matching images from a single user idea.

**Example**:

*   **Input**: "A superhero who dresses like an ant"
    
*   **Output**: 4-panel comic with text descriptions + 4 AI-generated images
    

⚡ Quick Start (3 Steps!)
------------------------

### Step 1: Install Dependencies

` pip install streamlit transformers torch diffusers accelerate safetensors pillow   `

Or use the requirements file:
 
`pip install -r requirements.txt`

### Step 2: Save the Code

Save the main code as comic\_with\_images.py

### Step 3: Run the App
 
`streamlit run comic_with_images.py`

The app will open in your browser at http://localhost:8501

📦 What Gets Downloaded (Automatic - First Run Only)
----------------------------------------------------

*   **GPT-2 model** (~500MB) - for text generation
    
*   **Stable Diffusion v1.4** (~4GB) - for image generation
    
*   **Total**: ~4.5GB
    
*   **Time**: 5-10 minutes first run, then cached permanently
    

⚠️ Make sure you have at least **10GB free disk space**

🎯 Two Generation Modes
-----------------------

### Mode 1: Template-Based (RECOMMENDED for demos)

*   ✅ **100% coherent stories** - guaranteed logical flow
    
*   ✅ **Fast** - instant text generation
    
*   ✅ **Reliable** - perfect for presentations
    
*   ✅ **Genre-specific** - comedy, adventure, sci-fi patterns
    
*   **Use this for**: Live demos, presentations, guaranteed results
    

### Mode 2: AI Generation

*   ✅ **More creative** - unique story variations
    
*   ⚠️ **Less predictable** - may produce disconnected panels
    
*   ✅ **Fallback safety** - uses templates if AI fails
    
*   **Use this for**: Experimentation, when you want surprise results
    

**Why Two Modes?**

GPT-2 (124M parameters) sometimes struggles with story coherence across 4 panels. Template mode guarantees quality, while AI mode attempts creative generation with a safety net.

💻 System Requirements
----------------------

### Minimum (CPU Mode):

*   **RAM**: 8GB
    
*   **Disk**: 10GB free
    
*   **Processor**: Multi-core CPU
    
*   **Speed**: 2-5 minutes per comic (text + images)
    

### Recommended (GPU Mode):

*   **RAM**: 16GB
    
*   **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1060 or better)
    
*   **Disk**: 10GB free
    
*   **Speed**: 30-60 seconds per comic
    

**Note**: The app automatically detects your hardware and optimizes accordingly!

🎯 How to Demo This
-------------------

### For Presentation (BEST STRATEGY):

**Option A - Safe & Fast (Recommended)**:

1.  Use **Template-Based mode** for guaranteed coherent stories
    
2.  **Disable images** during live demo (instant results!)
    
3.  Show **pre-generated examples** with images separately
    
4.  Explain: "Image generation takes 2-3 minutes on CPU, so here are examples I made earlier"
    

**Option B - Full Demo (If you have GPU)**:

1.  Use **Template-Based mode**
    
2.  Generate 1 complete comic live (30-60 seconds on GPU)
    
3.  Take audience suggestions for ideas
    
4.  Show different genres and art styles
    

**Option C - Hybrid (Best of Both)**:

1.  Start with text-only generation (instant, impressive)
    
2.  Show 2-3 pre-generated comics with images
    
3.  Explain the two-stage pipeline while generation runs in background
    

### Best Demo Ideas:

*   "A superhero who dresses like an ant"
    
*   "Programmer debugging code at 3 AM"
    
*   "Student trying to stay awake in class"
    
*   "Two friends fighting over last biryani"
    
*   "A cat who thinks it's a lion"
    

🎨 Features & Customization
---------------------------

### Story Settings:

*   **Generation Mode**: Template-Based vs AI Generation
    
*   **Genre**: comedy, adventure, sci-fi, romance, slice-of-life, horror-comedy
    
*   **Tone**: funny, wholesome, dramatic, sarcastic, absurd
    

### Image Settings:

*   **Art Style**: comic book, cartoon, anime, pixel art, watercolor, sketch
    
*   **Enable/Disable Images**: Toggle for speed
    
*   **Quality Control**: Automatic prompt cleaning and optimization
    

### User Interface:

*   2x2 grid layout for panels
    
*   Random idea generator
    
*   Example suggestions
    
*   Expandable raw output view
    
*   Tips and troubleshooting guide
    

🔧 Technical Architecture
-------------------------

### Two-Stage Pipeline:

` User Input → [Stage 1: Text Generation] → [Stage 2: Image Generation] → Display `

**Stage 1 - Text Generation (GPT-2)**:

*   Generates 4-panel story structure
    
*   Template mode OR AI generation
    
*   Automatic fallback for coherence
    
*   Output: 4 clean scene descriptions
    

**Stage 2 - Image Generation (Stable Diffusion)**:

*   Converts each text panel to image prompt
    
*   Cleans prompts (removes URLs, limits length)
    
*   Adds style keywords
    
*   Generates 512x512 images
    
*   Output: 4 matching illustrations