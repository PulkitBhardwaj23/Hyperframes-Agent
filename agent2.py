import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- UPDATED: New Langchain Ollama Import ---
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# --- 1. THE IMMUTABLE TEMPLATE ---
TEMPLATE_CODE = """<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; background-color: #0a0a0a; overflow: hidden; font-family: 'Helvetica Neue', sans-serif; }
        .bg-video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 0; }
        #vid1 { opacity: 0.4; } 
        #vid2 { opacity: 0; }   
        #vid3 { opacity: 0; }   
        #webgl-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }
        #ui-layer { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none; }
        #scene-1-ui { width: 50%; height: 100%; padding: 120px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; position: absolute; top: 0; left: 0; }
        #scene-2-ui { width: 50%; height: 100%; padding: 120px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; position: absolute; top: 0; left: 0; opacity: 0; }
        .s2-feature { margin-top: 30px; background: rgba(26, 26, 26, 0.8); padding: 20px 30px; border-left: 6px solid #00ffcc; width: fit-content; }
        #scene-3-ui { width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; padding-bottom: 150px; position: absolute; top: 0; left: 0; opacity: 0; box-sizing: border-box; }
        #cta-btn { background: white; color: black; font-size: 50px; font-weight: bold; padding: 30px 80px; border-radius: 100px; text-transform: uppercase; }
    </style>
    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
      }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js"></script>
</head>
<body>
    <div id="root" data-composition-id="threejs-split" data-width="1920" data-height="1080" data-duration="15">
        <video id="vid1" class="clip bg-video" data-start="0" data-duration="5" src="./clip.mp4" muted playsinline loop></video>
        <video id="vid2" class="clip bg-video" data-start="5" data-duration="5" src="./bg_action.mp4" muted playsinline loop></video>
        <video id="vid3" class="clip bg-video" data-start="10" data-duration="5" src="./bg_dark.mp4" muted playsinline loop></video>
        <canvas id="webgl-canvas"></canvas>
        <div id="ui-layer">
            <div id="scene-1-ui">
                <h1 id="title" style="color: white; font-size: 110px; margin: 0; line-height: 1.1;">[[S1_TITLE]]</h1>
                <p id="subtitle" style="color: #ccc; font-size: 40px; margin-top: 30px; max-width: 600px;">[[S1_SUB]]</p>
                <div id="feature-box"><h3 style="color: white; font-size: 35px; margin: 0;">[[S1_FEAT]]</h3></div>
            </div>
            <div id="scene-2-ui">
                <h1 style="color: white; font-size: 90px; margin: 0; line-height: 1.1;">[[S2_TITLE]]</h1>
                <div class="s2-feature"><h3 style="color: white; font-size: 30px; margin: 0;">[[S2_FEAT1]]</h3></div>
                <div class="s2-feature"><h3 style="color: white; font-size: 30px; margin: 0;">[[S2_FEAT2]]</h3></div>
            </div>
            <div id="scene-3-ui">
                <div id="cta-btn">[[S3_CTA]]</div>
            </div>
        </div>
    </div>
    <script type="module">
        import * as THREE from 'three';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
        const canvas = document.querySelector('#webgl-canvas');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, 1920/1080, 0.1, 100);
        camera.position.set(0, 0, 7); 
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        renderer.setSize(1920, 1080);
        scene.add(new THREE.AmbientLight(0xffffff, 1.2));
        const dirLight = new THREE.DirectionalLight(0xffffff, 2.5);
        dirLight.position.set(5, 5, 5);
        scene.add(dirLight);
        const tl = gsap.timeline({ paused: true });
        window.__timelines = { "threejs-split": tl };
        tl.set("#vid1", { opacity: 0.4 }, 0);
        tl.from("#title", { x: -100, opacity: 0, duration: 1, ease: "power2.out" }, 0.5);
        tl.from("#subtitle", { x: -50, opacity: 0, duration: 1, ease: "power2.out" }, 0.8);
        tl.from("#feature-box", { y: 50, opacity: 0, duration: 1, ease: "back.out(1.5)" }, 1.5);
        tl.to("#scene-1-ui", { opacity: 0, x: -50, duration: 0.5 }, 4.5);
        tl.to("#vid2", { opacity: 0.4, duration: 0.5 }, 5);
        tl.to("#scene-2-ui", { opacity: 1, duration: 0.5 }, 5);
        tl.from(".s2-feature", { x: -50, opacity: 0, duration: 0.8, stagger: 0.2, ease: "power2.out" }, 5.5);
        tl.to("#scene-2-ui", { opacity: 0, x: -50, duration: 0.5 }, 9.5);
        tl.to("#vid2", { opacity: 0, duration: 0.5 }, 10);
        tl.to("#vid3", { opacity: 0.8, duration: 0.5 }, 10);
        tl.to("#scene-3-ui", { opacity: 1, duration: 0.5 }, 10);
        tl.from("#cta-btn", { y: 50, opacity: 0, duration: 1, ease: "back.out(1.5)" }, 10.5);
        const loader = new GLTFLoader();
        loader.load('./product.glb', function(gltf) {
            const product = gltf.scene;
            scene.add(product);
            product.position.set(2.5, -0.5, 0);
            product.scale.set(0.2, 0.2, 0.2);
            tl.to(product.rotation, { y: Math.PI * 2, duration: 4, ease: "power1.inOut" }, 0.5);
            tl.to(product.position, { x: 0, y: 0, z: 2, duration: 1.5, ease: "power2.inOut" }, 5);
            tl.to(product.rotation, { y: Math.PI * 2.5, duration: 4, ease: "none" }, 5); 
            tl.call(() => {
                product.traverse((child) => {
                    if (child.isMesh) {
                        child.userData.origMat = child.material;
                        child.material = new THREE.MeshBasicMaterial({ color: 0x00ffcc, wireframe: true, transparent: true, opacity: 0.8 });
                    }
                });
            }, null, 5.5);
            tl.call(() => {
                product.traverse((child) => {
                    if (child.isMesh && child.userData.origMat) {
                        child.material = child.userData.origMat;
                    }
                });
            }, null, 10);
            tl.fromTo(product.position, 
                { x: 0, y: 5, z: 0 }, 
                { x: 0, y: -1.5, z: 0, duration: 1.5, ease: "bounce.out", immediateRender: false }, 
            10);
            tl.to(product.rotation, { y: Math.PI * 4, duration: 4, ease: "power1.inOut" }, 10);
        });
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>"""

# --- 2. DEFINE STATE & LIGHTWEIGHT LLM ---
class AgentState(TypedDict):
    product_info: str
    goal: str
    draft_json: dict
    is_valid: bool
    loop_count: int
    final_html: str

# UPDATED: Using the 1B model to prevent CUDA OOM crash
llm = ChatOllama(model="llama3.2:1b", temperature=0.1)

# --- 3. NODE DEFINITIONS ---

def copywriter_node(state: AgentState):
    print(f"✍️  [Node 1] Generating Script (Attempt {state['loop_count'] + 1})...")
    
    prompt = f"""You are a commercial video copywriter. 
    Product Info: {state['product_info']}
    Goal: {state['goal']}
    
    Write punchy text for a 3-scene video. Output ONLY a raw JSON object with these exact keys:
    {{
        "S1_TITLE": "2-3 words, use <br> for break",
        "S1_SUB": "1 short sentence",
        "S1_FEAT": "2 words max",
        "S2_TITLE": "2-3 words, use <br> for break",
        "S2_FEAT1": "2 words max",
        "S2_FEAT2": "2 words max",
        "S3_CTA": "Call to action text"
    }}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        draft = json.loads(match.group(0))
    except:
        draft = {} # Force a failure in validation if JSON breaks
        
    return {"draft_json": draft}

def validator_node(state: AgentState):
    print("🛡️  [Node 2] Fact-Checking & Grounding Logic...")
    draft = state['draft_json']
    
    # Required keys for the template
    required_keys = ["S1_TITLE", "S1_SUB", "S1_FEAT", "S2_TITLE", "S2_FEAT1", "S2_FEAT2", "S3_CTA"]
    has_keys = all(k in draft for k in required_keys)
    
    # Send it to the LLM to verify facts against the original product info
    prompt = f"""Original Product Info: {state['product_info']}
    Draft Script: {json.dumps(draft)}
    
    Analyze the draft. Does it make claims not present in the original info? 
    Respond with ONLY 'PASS' or 'FAIL'. Do not explain."""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    
    is_valid = True if ("PASS" in response and has_keys) else False
    
    if is_valid:
        print("✅ Logic AND Gate Passed: Structure Intact & Facts Grounded.")
    else:
        print("❌ Hallucination or Missing Keys detected. Sending back to copywriter.")
        
    return {"is_valid": is_valid, "loop_count": state['loop_count'] + 1}

def compiler_node(state: AgentState):
    print("⚙️  [Node 3] Deterministic Compiler Running...")
    html = TEMPLATE_CODE
    data = state['draft_json']
    
    # Strict string replacement. The template structure cannot be altered.
    html = html.replace("[[S1_TITLE]]", data.get("S1_TITLE", "Product Name"))
    html = html.replace("[[S1_SUB]]", data.get("S1_SUB", "Product description."))
    html = html.replace("[[S1_FEAT]]", data.get("S1_FEAT", "Feature 1"))
    html = html.replace("[[S2_TITLE]]", data.get("S2_TITLE", "Core Features"))
    html = html.replace("[[S2_FEAT1]]", data.get("S2_FEAT1", "Feature 2"))
    html = html.replace("[[S2_FEAT2]]", data.get("S2_FEAT2", "Feature 3"))
    html = html.replace("[[S3_CTA]]", data.get("S3_CTA", "PRE-ORDER NOW"))
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print("🚀 index.html securely overwritten!")
    return {"final_html": html}

# --- 4. ROUTING LOGIC ---
def routing_logic(state: AgentState):
    if state['is_valid'] or state['loop_count'] >= 3:
        return "compiler"
    return "copywriter"

# --- 5. BUILD GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("copywriter", copywriter_node)
workflow.add_node("validator", validator_node)
workflow.add_node("compiler", compiler_node)

workflow.set_entry_point("copywriter")
workflow.add_edge("copywriter", "validator")

workflow.add_conditional_edges("validator", routing_logic)
workflow.add_edge("compiler", END)

video_pipeline = workflow.compile()

# --- EXECUTION ---
if __name__ == "__main__":
    inputs = {
        "product_info": "Internal freewheels TFS-12 inner diameter 12mm outer diameter 35mm width 13mm. Compact design. Premium quality indexing freewheels.",
        "goal": "LinkedIn B2B video targeting mechanical engineers.",
        "loop_count": 0
    }
    
    video_pipeline.invoke(inputs)