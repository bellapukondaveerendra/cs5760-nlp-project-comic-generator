# 🎬 4-Panel Comic Script Generator - Setup Guide

## Quick Start (3 Steps!)

### Step 1: Install Dependencies
```bash
pip install streamlit transformers torch
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Save the Code
Save the main code as `comic_generator.py`

### Step 3: Run the App
```bash
streamlit run comic_generator.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📦 What Gets Downloaded
- **GPT-2 model** (~500MB) - downloads automatically on first run
- Takes 1-2 minutes first time, then cached locally

---

## 🎯 How to Demo This

### For Presentation:
1. **Start with a fun example**: "Two friends fighting over last biryani"
2. **Show genre switching**: Try comedy vs horror-comedy
3. **Take audience suggestions**: Let them give ideas live
4. **Show the variety**: Generate 2-3 different scripts

### Best Demo Ideas:
- College situations (relatable to judges)
- Food fights
- Exam prep struggles
- Coding bugs at 3 AM
- Family WhatsApp groups

---

## 🚀 Enhancement Ideas (If You Have Time)

### Easy Adds:
1. **Save/Export button** - Download script as text
2. **Multiple outputs** - Generate 3 versions, user picks best
3. **Character names** - Let user specify protagonist name

### Medium Adds:
4. **Telugu mode** - Generate in Telugu script
5. **Ratings system** - Users rate outputs (save to file)
6. **Image placeholders** - Add stick figure illustrations

### Advanced (Post-Demo):
7. **Fine-tune GPT-2** on Reddit jokes or comedy scripts
8. **Add DALL-E API** for actual comic panel generation
9. **Speech bubbles** - Format output as dialogue

---

## 🐛 Common Issues & Fixes

### Issue: "Model too slow"
**Fix**: You're already using GPT-2 (smallest). If still slow:
- Reduce `max_length` from 300 to 200
- Use `num_beams=1` in generation

### Issue: "Output is weird/incomplete"
**Fix**: Adjust generation parameters:
```python
temperature=0.9  # Higher = more creative
top_p=0.9        # Lower = more focused
```

### Issue: "Not funny enough"
**Fix**: 
- Add "very funny" or "hilarious" to prompt
- Try "absurd" tone
- Use specific humor styles in prompt

---

## 📊 What to Highlight in Presentation

### Technical Skills:
✅ Transformer models (GPT-2)
✅ Prompt engineering
✅ Text generation with constraints
✅ Parameter tuning (temperature, top_p)
✅ UI/UX with Streamlit

### NLP Concepts:
✅ Language modeling
✅ Conditional generation
✅ Structure in unstructured generation
✅ Creative AI applications

### Practical Value:
✅ Content creation tool
✅ Writer's block helper
✅ Social media content generator
✅ Educational tool for story structure

---

## 🎓 Explaining the Model

**Simple Explanation:**
"GPT-2 is trained on millions of internet texts. I give it a structured prompt with the 4-panel format, and it learns to fill in creative, coherent scenes that follow comic storytelling patterns."

**Technical Explanation:**
"Using autoregressive language modeling with GPT-2, I employ few-shot prompting to condition the model on the comic structure. Temperature sampling (0.8) balances creativity with coherence, while top-p nucleus sampling prevents repetitive outputs."

---

## 💾 File Structure
```
comic-generator/
├── comic_generator.py   # Main app
├── requirements.txt     # Dependencies
└── README.md           # This file
```

---

## ⚡ Quick Test Commands

Test in Python directly (no UI):
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

prompt = """Write a funny 4-panel comic:
Panel 1 (Setup): Student opens laptop to study
Panel 2 (Conflict):"""

output = generator(prompt, max_length=100, temperature=0.8)
print(output[0]['generated_text'])
```

---

## 🏆 Why This Project Stands Out

1. **Immediate entertainment value** - judges will laugh
2. **Shows creativity + technical skills** - not just classification
3. **Lightweight** - runs on any laptop
4. **Extensible** - clear path to improvements
5. **Unique** - not the usual sentiment analysis/chatbot

---

Good luck with your presentation! 🎉