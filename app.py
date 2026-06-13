import streamlit as st
import os
import json
import re
import subprocess
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ==========================================
# 1. DIRECTORY SETUP
# ==========================================
TEMPLATES_DIR = "templates" 
AUDIO_DIR = "audio"
OUTPUT_DIR = "output"

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Retrieve API key dynamically
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=8000, api_key=GROQ_API_KEY)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_catalog():
    """Scans the /template folder and builds a JSON catalog for the LLM."""
    catalog = []
    if not os.path.exists(TEMPLATES_DIR):
        return catalog
        
    for folder in os.listdir(TEMPLATES_DIR):
        config_path = os.path.join(TEMPLATES_DIR, folder, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    catalog.append({
                        "id": folder,
                        "name": data.get("name", folder),
                        "tags": data.get("tags", "")
                    })
                except json.JSONDecodeError:
                    pass
    return catalog

def get_template_config(template_id):
    """Loads the specific config for the chosen template."""
    config_path = os.path.join(TEMPLATES_DIR, template_id, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_template_html(template_id):
    """Loads the raw HTML for the chosen template."""
    html_path = os.path.join(TEMPLATES_DIR, template_id, "template.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_html(text: str) -> str:
    """Safely extracts HTML from LLM output."""
    if not text: return ""
    start_idx = text.lower().find("<!doctype")
    if start_idx == -1: start_idx = text.lower().find("<html")
    if start_idx != -1:
        end_idx = text.lower().rfind("</html>")
        if end_idx != -1: return text[start_idx:end_idx + 7]
    fence = "`" * 3
    pattern = fence + r"(?:html)?\s*(.*?)\s*" + fence
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return text.strip()

def get_audio_files():
    """Lists all .mp3 files in the audio folder."""
    if not os.path.exists(AUDIO_DIR):
        return []
    return [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]

# ==========================================
# 3. LANGGRAPH ARCHITECTURE
# ==========================================
class GraphState(TypedDict):
    product_info: str
    constraints: str
    catalog: str
    selected_template_id: str
    base_html: str
    final_html: str

def select_design_node(state: GraphState) -> GraphState:
    prompt = f"""
    You are an Art Director. Choose the best template ID from this catalog.
    
    CATALOG:
    {state['catalog']}
    
    PRODUCT INFO: {state['product_info']}
    USER CONSTRAINTS (Duration, Size, Vibe): {state['constraints']}
    
    Output ONLY the exact 'id' string of the best template. Do not explain.
    """
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '', response)
    
    return {**state, "selected_template_id": clean_id}

def fit_content_node(state: GraphState) -> GraphState:
    # UPDATED STRICT PROMPT
    prompt = f"""
    You are an expert Copywriter and Frontend Developer.
    PRODUCT INFO: {state['product_info']}
    
    INSTRUCTIONS: 
    1. Replace EVERY single placeholder in brackets like [BRAND_NAME], [SCENE_1_TITLE], [SCENE_3_SPEC_1_LABEL] inside the BASE TEMPLATE with real ad copy based on the PRODUCT INFO.
    2. For placeholders ending in _ICON (e.g., [SCENE_3_SPEC_1_ICON]), use exactly ONE relevant emoji (e.g., 🚀, 🛡️, ⚡).
    3. DO NOT alter any HTML tags, CSS, GSAP scripts, or `src="..."` attributes. Leave media alone.
    4. CRITICAL: Output the entire, fully complete HTML code. Do NOT truncate. Do NOT add explanations.
    
    BASE TEMPLATE:
    ```html
    {state['base_html']}
    ```
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        clean_html = extract_html(response.content)
        
        # STRICT VALIDATION: If the AI truncates, we throw a hard error instead of using the blank template.
        if len(clean_html) < len(state['base_html']) * 0.5:
            return {**state, "final_html": "ERROR: The AI generated truncated code. Please click Compile again."}
            
    except Exception as e:
        return {**state, "final_html": f"ERROR: API Connection Failed. {str(e)}"}
        
    return {**state, "final_html": clean_html}

workflow = StateGraph(GraphState)
workflow.add_node("selector", select_design_node)
workflow.add_node("fitter", fit_content_node)
workflow.set_entry_point("selector")
workflow.add_edge("selector", "fitter")
workflow.add_edge("fitter", END)
ad_agent = workflow.compile()

# ==========================================
# 4. STREAMLIT UI (Two-Phase System)
# ==========================================
st.set_page_config(page_title="Hyperframes Engine", page_icon="⚙️", layout="centered")

if "phase" not in st.session_state:
    st.session_state.phase = 1
if "selected_template_id" not in st.session_state:
    st.session_state.selected_template_id = ""
if "product_info" not in st.session_state:
    st.session_state.product_info = ""

st.title("⚙️ Hyperframes Ad Engine")

# ---------------------------------------------------------
# PHASE 1: The Briefing (Strictly Text & Selection)
# ---------------------------------------------------------
if st.session_state.phase == 1:
    st.write("### Step 1: Campaign Brief")
    
    product_info = st.text_area("📦 Product Description", height=150)
    constraints = st.text_input("🎯 Video Requirements (e.g., '15s portrait social media')")
    
    if st.button("Analyze & Select Design", use_container_width=True):
        catalog_list = load_catalog()
        if not catalog_list:
            st.error("⚠️ No templates found in the /template folder!")
        elif not product_info:
            st.error("⚠️ Please enter the product description.")
        else:
            with st.spinner("🤖 AI is analyzing the catalog for the best fit..."):
                state = {
                    "product_info": product_info, 
                    "constraints": constraints, 
                    "catalog": json.dumps(catalog_list), 
                    "selected_template_id": "", 
                    "base_html": "", 
                    "final_html": ""
                }
                result = select_design_node(state)
                
                selected_id = result["selected_template_id"]
                
                if os.path.exists(os.path.join(TEMPLATES_DIR, selected_id)):
                    st.session_state.selected_template_id = selected_id
                    st.session_state.product_info = product_info
                    st.session_state.phase = 2
                    st.rerun()
                else:
                    st.error(f"⚠️ AI selected '{selected_id}', but that folder doesn't exist.")

# ---------------------------------------------------------
# PHASE 2: Dynamic Asset Upload & Compilation
# ---------------------------------------------------------
elif st.session_state.phase == 2:
    st.write("### Step 2: Upload Required Assets")
    
    t_id = st.session_state.selected_template_id
    config = get_template_config(t_id)
    
    st.success(f"🎯 AI Selected Design: **{config.get('name', t_id)}**")
    
    uploaded_assets = {}
    st.write("#### Required Visuals")
    for asset in config.get("required_assets", []):
        file = st.file_uploader(asset["prompt"], type=asset["type"], key=asset["name"])
        uploaded_assets[asset["name"]] = file
    
    st.write("#### Audio Soundtrack")
    available_audio = get_audio_files()
    if available_audio:
        selected_audio = st.selectbox("Select Background Track", ["None"] + available_audio)
    else:
        st.warning("No .mp3 files found in /audio folder.")
        selected_audio = "None"
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Step 1"):
            st.session_state.phase = 1
            st.rerun()
            
    with col2:
        if st.button("🚀 Auto-Compile Final Video", use_container_width=True):
            missing_files = [k for k, v in uploaded_assets.items() if v is None]
            if missing_files:
                st.error(f"⚠️ Please upload missing files: {', '.join(missing_files)}")
            else:
                with st.spinner("🤖 1/3: AI is writing the GSAP/HTML Code..."):
                    # Save user media directly to the output folder
                    for file_name, file_obj in uploaded_assets.items():
                        with open(os.path.join(OUTPUT_DIR, file_name), "wb") as f:
                            f.write(file_obj.getbuffer())
                            
                    # Trigger the LLM to write the copy
                    base_html = get_template_html(t_id)
                    state = {
                        "product_info": st.session_state.product_info,
                        "constraints": "", 
                        "catalog": "", 
                        "selected_template_id": t_id,
                        "base_html": base_html, 
                        "final_html": ""
                    }
                    result = fit_content_node(state)
                    
                    # --- THE HARD STOP SAFETY NET ---
                    # If the AI failed, stop the app immediately. No broken renders.
                    if result["final_html"].startswith("ERROR:"):
                        st.error(result["final_html"])
                        st.stop()
                    
                    # Save the successfully injected HTML to the output folder
                    final_html_path = os.path.join(OUTPUT_DIR, "index.html")
                    with open(final_html_path, "w", encoding="utf-8") as f:
                        f.write(result["final_html"])
                
                with st.spinner("🎥 2/3: Hyperframes is rendering the visual frames... (This takes a minute)"):
                    render_cmd = "npx hyperframes render . --output silent_ad.mp4"
                    subprocess.run(render_cmd, shell=True, cwd=OUTPUT_DIR)
                
                with st.spinner("🎵 3/3: Stitching Audio Track..."):
                    if selected_audio != "None":
                        audio_path = os.path.abspath(os.path.join(AUDIO_DIR, selected_audio))
                        silent_vid = os.path.abspath(os.path.join(OUTPUT_DIR, "silent_ad.mp4"))
                        final_vid = os.path.abspath(os.path.join(OUTPUT_DIR, "final_ad_with_audio.mp4"))
                        
                        ffmpeg_cmd = [
                            "ffmpeg", "-y", 
                            "-i", silent_vid, 
                            "-i", audio_path, 
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-shortest",
                            final_vid
                        ]
                        
                        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        
                        if process.returncode == 0:
                            st.success("✅ Compilation Complete! Watch your final video below:")
                            st.video(final_vid)
                        else:
                            st.error("⚠️ FFmpeg failed to stitch the audio.")
                            with st.expander("Show FFmpeg Error Log"):
                                st.code(process.stderr)
                    else:
                        silent_vid = os.path.abspath(os.path.join(OUTPUT_DIR, "silent_ad.mp4"))
                        st.success("✅ Compilation Complete! Watch your final video below:")
                        st.video(silent_vid)