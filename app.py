import streamlit as st
import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ==========================================
# 1. TEMPLATES 
# ==========================================
TEMPLATE_A_ORBS = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; padding: 0; background-color: #020204; color: white; overflow: hidden; font-family: 'Roboto', sans-serif; }
    #root { position: relative; width: 1080px; height: 1920px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #0a0a14 0%, #020204 100%); }
    .bg-orb { position: absolute; border-radius: 50%; filter: blur(150px); z-index: 0; opacity: 0.5; }
    #orb-cyan { width: 1000px; height: 1000px; background: #00FFFF; top: -300px; left: -300px; }
    #orb-gold { width: 1200px; height: 1200px; background: #FFD700; bottom: -400px; right: -400px; }
    .brand-logo { position: absolute; top: 60px; right: 60px; font-family: 'Montserrat'; font-weight: 900; font-size: 45px; color: #fff; letter-spacing: 2px; z-index: 10; text-shadow: 0 0 20px rgba(255,255,255,0.3); }
    .corner-accent { position: absolute; width: 150px; height: 150px; border: 4px solid transparent; z-index: 10; }
    .corner-top-left { top: 50px; left: 50px; border-top-color: #00FFFF; border-left-color: #00FFFF; }
    .corner-bottom-right { bottom: 50px; right: 50px; border-bottom-color: #FFD700; border-right-color: #FFD700; }
    .scene { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; opacity: 0; text-align: center; padding: 0 8%; z-index: 10; }
    .text-small-accent { font-family: 'Montserrat'; font-weight: 700; font-size: 35px; color: #FFD700; letter-spacing: 8px; text-transform: uppercase; margin-bottom: 25px; }
    .text-giant { font-family: 'Montserrat'; font-weight: 900; font-size: 140px; color: #FFFFFF; line-height: 1.05; text-transform: uppercase; text-shadow: 0 0 10px rgba(0, 255, 255, 0.2); margin-bottom: 40px; }
    .text-cyan { color: #00FFFF; }
    .text-subtitle { font-family: 'Roboto'; font-weight: 400; font-size: 45px; color: #d1d1d1; letter-spacing: 3px; }
    .image-box { position: relative; width: 750px; height: 850px; background: #0a0a0a; border-radius: 30px; padding: 40px; display: flex; justify-content: center; align-items: center; overflow: hidden; margin-bottom: 60px; border: 3px solid rgba(0, 255, 255, 0.3); box-shadow: 0 0 20px rgba(0, 255, 255, 0.1), inset 0 0 20px rgba(0, 255, 255, 0.1); }
    .image-box img { max-width: 100%; max-height: 100%; object-fit: contain; z-index: 2; position: relative; }
    .cta-btn { margin-top: 60px; padding: 35px 90px; background: transparent; border: 5px solid #FFD700; color: #FFD700; font-family: 'Montserrat'; font-weight: 900; font-size: 55px; border-radius: 100px; text-transform: uppercase; letter-spacing: 4px; box-shadow: 0 0 20px rgba(255, 215, 0, 0.2); }
</style>
<script src="https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js"></script>
</head>
<body>
<div id="root" data-composition-id="main-composition" data-width="1080" data-height="1920" data-duration="15">
    <div id="orb-cyan" class="bg-orb"></div><div id="orb-gold" class="bg-orb"></div>
    <div class="brand-logo">[BRAND_NAME]</div>
    <div class="corner-accent corner-top-left"></div><div class="corner-accent corner-bottom-right"></div>
    <div id="scene1" class="scene">
        <div class="s1-item text-small-accent">[SCENE_1_SUBTITLE]</div>
        <div class="s1-item text-giant text-cyan">[SCENE_1_TITLE]</div>
        <div class="s1-item text-subtitle">[SCENE_1_DESC]</div>
    </div>
    <div id="scene2" class="scene">
        <div class="image-box s2-img"><img src="product.jpeg" alt="Product Image 1"></div>
        <div class="s2-text text-giant" style="font-size: 100px; margin-bottom: 20px;">[SCENE_2_TITLE]</div>
        <div class="s2-text text-subtitle">[SCENE_2_DESC]</div>
    </div>
    <div id="scene3" class="scene">
        <div class="s3-item text-giant">[SCENE_3_TITLE]</div>
        <div class="s3-item text-subtitle" style="font-size: 40px; line-height: 1.5; color: #aaa;">
            [SCENE_3_DESC_LINE_1]<br><span style="color:#FFF; font-weight:700;">[SCENE_3_SPEC_1]</span><br><span style="color:#FFF; font-weight:700;">[SCENE_3_SPEC_2]</span>
        </div>
    </div>
    <div id="scene4" class="scene">
        <div class="image-box s4-img"><img src="product2.jpeg" alt="Product Image 2"></div>
        <div class="s4-text text-small-accent" style="color: #00FFFF;">[SCENE_4_SUBTITLE]</div>
        <div class="s4-text text-giant" style="font-size: 110px; margin-bottom: 20px;">[SCENE_4_TITLE]</div>
        <div class="s4-text text-subtitle" style="font-size: 38px;">[SCENE_4_DESC]</div>
    </div>
    <div id="scene5" class="scene">
        <div class="s5-item text-small-accent">[SCENE_5_SUBTITLE]</div>
        <div class="s5-item text-giant" style="font-size: 120px; color: #00FFFF;">[SCENE_5_TITLE]</div>
        <div class="s5-item cta-btn">ENQUIRE NOW</div>
    </div>
</div>
<script>
    const tl = gsap.timeline({ paused: true });
    window.__timelines = { "main-composition": tl };
    tl.to('#orb-cyan', { x: 600, y: 500, duration: 15, ease: "sine.inOut" }, 0);
    tl.to('#orb-gold', { x: -700, y: -600, duration: 15, ease: "sine.inOut" }, 0);
    tl.to('.image-box', { boxShadow: "0 0 80px rgba(0, 255, 255, 0.7), inset 0 0 40px rgba(0, 255, 255, 0.4)", borderColor: "rgba(0, 255, 255, 1)", duration: 1.5, yoyo: true, repeat: 9, ease: "sine.inOut" }, 0);
    tl.to('.text-giant', { textShadow: "0 0 30px rgba(0, 255, 255, 0.8), 0 0 60px rgba(0, 255, 255, 0.5)", duration: 1.5, yoyo: true, repeat: 9, ease: "sine.inOut" }, 0);
    tl.to('.cta-btn', { boxShadow: "0 0 60px rgba(255, 215, 0, 0.8)", borderColor: "rgba(255, 215, 0, 1)", duration: 1.5, yoyo: true, repeat: 9, ease: "sine.inOut" }, 0);
    gsap.set('.scene', { opacity: 0 });
    tl.to('#scene1', { opacity: 1, duration: 0.4 }, 0).from('.s1-item', { scale: 0.9, opacity: 0, duration: 1.2, stagger: 0.2, ease: "power2.out" }, 0.2).to('#scene1', { opacity: 0, duration: 0.4 }, 2.8);
    tl.to('#scene2', { opacity: 1, duration: 0.4 }, 3).from('.s2-img', { scale: 0.85, opacity: 0, y: 30, duration: 0.9, ease: "back.out(1.2)" }, 3.2).from('.s2-text', { y: 20, opacity: 0, duration: 0.7, stagger: 0.15, ease: "power2.out" }, 3.6).to('#scene2', { opacity: 0, duration: 0.4 }, 6.2);
    tl.to('#scene3', { opacity: 1, duration: 0.4 }, 6.5).from('.s3-item', { scale: 1.05, filter: 'blur(5px)', opacity: 0, duration: 0.8, stagger: 0.2, ease: "power2.out" }, 6.7).to('.s3-item', { scale: 1.02, duration: 2.5, ease: "none" }, 7).to('#scene3', { opacity: 0, duration: 0.4 }, 9.2);
    tl.to('#scene4', { opacity: 1, duration: 0.4 }, 9.5).from('.s4-img', { scale: 0.85, opacity: 0, y: -30, duration: 0.9, ease: "back.out(1.2)" }, 9.7).from('.s4-text', { y: 20, opacity: 0, duration: 0.7, stagger: 0.15, ease: "power2.out" }, 10.1).to('#scene4', { opacity: 0, duration: 0.4 }, 12.5);
    tl.to('#scene5', { opacity: 1, duration: 0.4 }, 12.8).from('.s5-item', { y: 30, opacity: 0, duration: 0.8, stagger: 0.2, ease: "power3.out" }, 13.0).to('.cta-btn', { scale: 1.05, boxShadow: "0 0 60px rgba(255, 215, 0, 0.5)", duration: 0.6, yoyo: true, repeat: 2, ease: "sine.inOut" }, 14.0);
    tl.play();
</script>
</body>
</html>"""

TEMPLATE_B_PARTICLES = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; padding: 0; background-color: #020612; color: white; overflow: hidden; font-family: 'Roboto', sans-serif; }
    #root { position: relative; width: 1080px; height: 1920px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #061536 0%, #020612 80%); }
    #moving-grid { position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background-image: linear-gradient(rgba(0, 255, 255, 0.08) 2px, transparent 2px), linear-gradient(90deg, rgba(0, 255, 255, 0.08) 2px, transparent 2px); background-size: 150px 150px; z-index: 0; opacity: 0.6; }
    #particle-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }
    .glowing-dot { position: absolute; border-radius: 50%; }
    .brand-logo { position: absolute; top: 60px; right: 60px; font-family: 'Montserrat'; font-weight: 900; font-size: 45px; color: #fff; letter-spacing: 2px; z-index: 10; }
    .corner-accent { position: absolute; width: 150px; height: 150px; border: 4px solid transparent; z-index: 10; }
    .corner-top-left { top: 50px; left: 50px; border-top-color: #00FFFF; border-left-color: #00FFFF; }
    .corner-bottom-right { bottom: 50px; right: 50px; border-bottom-color: #FFD700; border-right-color: #FFD700; }
    .scene { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; opacity: 0; text-align: center; padding: 0 8%; z-index: 10; }
    .text-small-accent { font-family: 'Montserrat'; font-weight: 700; font-size: 35px; color: #FFD700; letter-spacing: 8px; text-transform: uppercase; margin-bottom: 25px; }
    .text-giant { font-family: 'Montserrat'; font-weight: 900; font-size: 140px; color: #FFFFFF; line-height: 1.05; text-transform: uppercase; text-shadow: 0 15px 40px rgba(0,0,0,0.8); margin-bottom: 40px; }
    .text-cyan { color: #00FFFF; }
    .text-subtitle { font-family: 'Roboto'; font-weight: 400; font-size: 45px; color: #a4c0eb; letter-spacing: 3px; }
    .image-box { position: relative; width: 750px; height: 850px; background: #ffffff; border-radius: 30px; padding: 40px; display: flex; justify-content: center; align-items: center; box-shadow: 0 30px 60px rgba(0,0,0,0.6); overflow: hidden; margin-bottom: 60px; }
    .image-box img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .cta-btn { margin-top: 60px; padding: 35px 90px; background: transparent; border: 5px solid #FFD700; color: #FFD700; font-family: 'Montserrat'; font-weight: 900; font-size: 55px; border-radius: 100px; text-transform: uppercase; letter-spacing: 4px; box-shadow: 0 0 40px rgba(255, 215, 0, 0.15); }
</style>
<script src="https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js"></script>
</head>
<body>
<div id="root" data-composition-id="main-composition" data-width="1080" data-height="1920" data-duration="15">
    <div id="moving-grid"></div><div id="particle-container"></div>
    <div class="brand-logo">[BRAND_NAME]</div>
    <div class="corner-accent corner-top-left"></div><div class="corner-accent corner-bottom-right"></div>
    <div id="scene1" class="scene">
        <div class="s1-item text-small-accent">[SCENE_1_SUBTITLE]</div>
        <div class="s1-item text-giant text-cyan">[SCENE_1_TITLE]</div>
        <div class="s1-item text-subtitle">[SCENE_1_DESC]</div>
    </div>
    <div id="scene2" class="scene">
        <div class="image-box s2-img"><img src="product.jpeg" alt="Product Image"></div>
        <div class="s2-text text-giant" style="font-size: 100px; margin-bottom: 20px;">[SCENE_2_TITLE]</div>
        <div class="s2-text text-subtitle">[SCENE_2_DESC]</div>
    </div>
    <div id="scene3" class="scene">
        <div class="s3-item text-giant">[SCENE_3_TITLE]</div>
        <div class="s3-item text-subtitle" style="font-size: 40px; line-height: 1.5; color: #a4c0eb;">
            [SCENE_3_DESC_LINE_1]<br><span style="color:#FFF; font-weight:700;">[SCENE_3_SPEC_1]</span><br><span style="color:#FFF; font-weight:700;">[SCENE_3_SPEC_2]</span>
        </div>
    </div>
    <div id="scene4" class="scene">
        <div class="image-box s4-img"><img src="product2.jpeg" alt="Product Image 2"></div>
        <div class="s4-text text-small-accent" style="color: #00FFFF;">[SCENE_4_SUBTITLE]</div>
        <div class="s4-text text-giant" style="font-size: 110px; margin-bottom: 20px;">[SCENE_4_TITLE]</div>
        <div class="s4-text text-subtitle" style="font-size: 38px;">[SCENE_4_DESC]</div>
    </div>
    <div id="scene5" class="scene">
        <div class="s5-item text-small-accent">[SCENE_5_SUBTITLE]</div>
        <div class="s5-item text-giant" style="font-size: 120px; color: #00FFFF;">[SCENE_5_TITLE]</div>
        <div class="s5-item cta-btn">ENQUIRE NOW</div>
    </div>
</div>
<script>
    const tl = gsap.timeline({ paused: true });
    window.__timelines = { "main-composition": tl };
    tl.to('#moving-grid', { x: -300, y: -300, duration: 15, ease: "none" }, 0);
    const particleContainer = document.getElementById('particle-container');
    const colors = ['#00FFFF', '#FFD700', '#ffffff'];
    for (let i = 0; i < 80; i++) {
        let dot = document.createElement('div');
        dot.classList.add('glowing-dot');
        particleContainer.appendChild(dot);
        let startX = Math.random() * 1080;
        let startY = Math.random() * 2400 - 200; 
        let size = Math.random() * 8 + 3; 
        let color = colors[Math.floor(Math.random() * colors.length)];
        gsap.set(dot, { x: startX, y: startY, width: size, height: size, backgroundColor: color, boxShadow: `0 0 ${size * 3}px ${color}`, opacity: Math.random() * 0.5 + 0.1 });
        let endY = startY - (Math.random() * 1000 + 500); 
        let endX = startX + (Math.random() * 400 - 200);
        tl.to(dot, { x: endX, y: endY, opacity: Math.random() * 0.9 + 0.3, ease: "sine.inOut", duration: 15 }, 0);
    }
    gsap.set('.scene', { opacity: 0 });
    tl.to('#scene1', { opacity: 1, duration: 0.4 }, 0).from('.s1-item', { scale: 0.9, opacity: 0, duration: 1.2, stagger: 0.2, ease: "power2.out" }, 0.2).to('#scene1', { opacity: 0, duration: 0.4 }, 2.8);
    tl.to('#scene2', { opacity: 1, duration: 0.4 }, 3).from('.s2-img', { scale: 0.85, opacity: 0, y: 30, duration: 0.9, ease: "back.out(1.2)" }, 3.2).from('.s2-text', { y: 20, opacity: 0, duration: 0.7, stagger: 0.15, ease: "power2.out" }, 3.6).to('#scene2', { opacity: 0, duration: 0.4 }, 6.2);
    tl.to('#scene3', { opacity: 1, duration: 0.4 }, 6.5).from('.s3-item', { scale: 1.05, filter: 'blur(5px)', opacity: 0, duration: 0.8, stagger: 0.2, ease: "power2.out" }, 6.7).to('.s3-item', { scale: 1.02, duration: 2.5, ease: "none" }, 7).to('#scene3', { opacity: 0, duration: 0.4 }, 9.2);
    tl.to('#scene4', { opacity: 1, duration: 0.4 }, 9.5).from('.s4-img', { scale: 0.85, opacity: 0, y: -30, duration: 0.9, ease: "back.out(1.2)" }, 9.7).from('.s4-text', { y: 20, opacity: 0, duration: 0.7, stagger: 0.15, ease: "power2.out" }, 10.1).to('#scene4', { opacity: 0, duration: 0.4 }, 12.5);
    tl.to('#scene5', { opacity: 1, duration: 0.4 }, 12.8).from('.s5-item', { y: 30, opacity: 0, duration: 0.8, stagger: 0.2, ease: "power3.out" }, 13.0).to('.cta-btn', { scale: 1.05, boxShadow: "0 0 60px rgba(255, 215, 0, 0.5)", duration: 0.6, yoyo: true, repeat: 2, ease: "sine.inOut" }, 14.0);
    tl.play();
</script>
</body>
</html>"""

# ==========================================
# 2. LANGGRAPH SETUP
# ==========================================
class GraphState(TypedDict):
    product_info: str
    selected_template_name: str
    base_html: str
    final_html: str

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=8000, api_key=GROQ_API_KEY)

def extract_html(text: str) -> str:
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

def select_design_node(state: GraphState) -> GraphState:
    prompt = f"""
    You are an Art Director choosing a visual template.
    Product Info: {state['product_info']}
    Option A: "TEMPLATE_A_ORBS" - Elegant, cinematic, ambient glowing orbs.
    Option B: "TEMPLATE_B_PARTICLES" - Technical, high-speed, moving grid with blue background and hovering particles.
    Output ONLY the string "TEMPLATE_A_ORBS" or "TEMPLATE_B_PARTICLES". Do not explain.
    """
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if "TEMPLATE_A" in response:
        return {**state, "selected_template_name": "TEMPLATE_A", "base_html": TEMPLATE_A_ORBS}
    else:
        return {**state, "selected_template_name": "TEMPLATE_B", "base_html": TEMPLATE_B_PARTICLES}

def fit_content_node(state: GraphState) -> GraphState:
    prompt = f"""
    You are an expert Frontend Developer.
    PRODUCT INFO: {state['product_info']}
    INSTRUCTIONS: 
    1. Replace the placeholders like [BRAND_NAME], [SCENE_1_TITLE], etc. in the BASE TEMPLATE with punchy ad copy. 
    2. Leave `<img src="product.jpeg"` and `<img src="product2.jpeg"` alone. Do not change image tags.
    3. DO NOT TRUNCATE. Output the entire HTML from <!DOCTYPE html> to </html>.
    BASE TEMPLATE:
    ```html
    {state['base_html']}
    ```
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        clean_html = extract_html(response.content)
        if len(clean_html) < 500: clean_html = state['base_html']
    except Exception:
        clean_html = state['base_html']
    return {**state, "final_html": clean_html}

workflow = StateGraph(GraphState)
workflow.add_node("selector", select_design_node)
workflow.add_node("fitter", fit_content_node)
workflow.set_entry_point("selector")
workflow.add_edge("selector", "fitter")
workflow.add_edge("fitter", END)
ad_agent = workflow.compile()

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Hyperframes Ad Generator", page_icon="🎬", layout="centered")

st.title("🎬 Hyperframes Ad Generator")
st.write("Upload your product details and images, and the AI will generate the full Hyperframes HTML code.")

user_product_info = st.text_area("📦 Product Description & Marketing Copy", height=150, placeholder="Paste details here...")

col1, col2 = st.columns(2)
with col1:
    prod_img1 = st.file_uploader("🖼️ Upload Product Image 1", type=['jpg', 'jpeg', 'png'])
with col2:
    prod_img2 = st.file_uploader("🖼️ Upload Product Image 2", type=['jpg', 'jpeg', 'png'])

if st.button("🚀 Generate Video HTML", use_container_width=True):
    if not user_product_info or not prod_img1 or not prod_img2:
        st.error("⚠️ Please provide the product description and upload BOTH images.")
    else:
        # Save the uploaded images locally as exactly product.jpeg and product2.jpeg
        with open("product.jpeg", "wb") as f:
            f.write(prod_img1.getbuffer())
        with open("product2.jpeg", "wb") as f:
            f.write(prod_img2.getbuffer())
            
        with st.spinner("🤖 AI Art Director & Developer are building your ad..."):
            result = ad_agent.invoke({
                "product_info": user_product_info,
                "selected_template_name": "",
                "base_html": "",
                "final_html": ""
            })
            
            final_code = result["final_html"]
            
            with open("index.html", "w", encoding="utf-8") as f:
                f.write(final_code)
                
            st.success(f"✅ Code Generated! (Design chosen: {result['selected_template_name']})")
            
            st.download_button(
                label="📥 Download index.html",
                data=final_code,
                file_name="index.html",
                mime="text/html",
                use_container_width=True
            )
            
            with st.expander("👀 View Generated HTML Code"):
                st.code(final_code, language='html')