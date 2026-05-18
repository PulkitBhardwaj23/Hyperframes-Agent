import sys
import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Updated import to use the dedicated Ollama package (fixes the warning)
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

sys.stdout.reconfigure(encoding='utf-8')

# 1. Define the State
class AgentState(TypedDict):
    raw_product_info: str
    script_json: dict
    final_html: str

# 2. Node 1: The Copywriter Agent
def creative_director_node(state: AgentState):
    print("🎬 [Node 1] Creative Director is writing the script...")
    
    # ENFORCING JSON MODE: format="json" stops Llama from outputting markdown or conversational text
    llm = ChatOllama(model="llama3", temperature=0.3, format="json") 
    
    prompt = f"""You are an elite commercial copywriter. 
    Turn the following technical product description into punchy, high-energy text for a 15-second video ad.
    
    Product Info: {state['raw_product_info']}
    
    Respond strictly with a JSON object matching this exact structure:
    {{
        "scene1_title": "2-3 words (use <br> for line breaks)",
        "scene1_subtitle": "One short punchy sentence.",
        "scene1_feature": "2-3 words",
        "scene2_title": "2-3 words (use <br> for line breaks)",
        "scene2_feature1": "2-3 words",
        "scene2_feature2": "2-3 words",
        "scene3_cta": "PRE-ORDER NOW or similar"
    }}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Safely parse the strict JSON
    try:
        script_data = json.loads(response.content)
        print("✅ Script generated successfully!")
    except Exception as e:
        # Fallback to regex if Llama somehow still outputs markdown
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            script_data = json.loads(json_match.group(0))
            print("✅ Script rescued using Regex!")
        except Exception as e2:
            print("❌ LLM failed to output valid JSON. Using fallback data.")
            print(f"Raw Output from LLM was:\n{response.content}")
            script_data = {
                "scene1_title": "System<br>Error", "scene1_subtitle": "LLM JSON parsing failed.",
                "scene1_feature": "Error", "scene2_title": "Data<br>Missing",
                "scene2_feature1": "Error", "scene2_feature2": "Error", "scene3_cta": "RETRY"
            }
        
    return {"script_json": script_data}

# 3. Node 2: The Compiler Agent
def compiler_node(state: AgentState):
    print("⚙️ [Node 2] Compiler is injecting script into HTML template...")
    data = state['script_json']
    
    # Your exact working HTML template
    html_template = """
<!DOCTYPE html>
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
                <p id="subtitle" style="color: #ccc; font-size: 40px; margin-top: 30px; max-width: 600px;">[[S1_SUBTITLE]]</p>
                <div id="feature-box"><h3 style="color: white; font-size: 35px; margin: 0;">[[S1_FEATURE]]</h3></div>
            </div>
            <div id="scene-2-ui">
                <h1 style="color: white; font-size: 90px; margin: 0; line-height: 1.1;">[[S2_TITLE]]</h1>
                <div class="s2-feature"><h3 style="color: white; font-size: 30px; margin: 0;">[[S2_FEATURE1]]</h3></div>
                <div class="s2-feature"><h3 style="color: white; font-size: 30px; margin: 0;">[[S2_FEATURE2]]</h3></div>
            </div>
            <div id="scene-3-ui">
                <div id="cta-btn">[[S3_CTA]]</div>
            </div>
        </div>
    </div>
    <script type="module">
        // ... (Your identical Three.js and GSAP logic remains here untouched)
    </script>
</body>
</html>
"""
    # Replace the placeholders with the LLM generated text
    final_html = html_template.replace("[[S1_TITLE]]", data.get("scene1_title", ""))
    final_html = final_html.replace("[[S1_SUBTITLE]]", data.get("scene1_subtitle", ""))
    final_html = final_html.replace("[[S1_FEATURE]]", data.get("scene1_feature", ""))
    final_html = final_html.replace("[[S2_TITLE]]", data.get("scene2_title", ""))
    final_html = final_html.replace("[[S2_FEATURE1]]", data.get("scene2_feature1", ""))
    final_html = final_html.replace("[[S2_FEATURE2]]", data.get("scene2_feature2", ""))
    final_html = final_html.replace("[[S3_CTA]]", data.get("scene3_cta", ""))
    
    # Save the new file
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print("✅ index.html overwritten successfully!")
    return {"final_html": final_html}

# 4. Build the Graph Workflow
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("copywriter", creative_director_node)
workflow.add_node("compiler", compiler_node)

# Define the flow
workflow.set_entry_point("copywriter")
workflow.add_edge("copywriter", "compiler")
workflow.add_edge("compiler", END)

# Compile the Graph
video_pipeline = workflow.compile()

# --- EXECUTION ---
if __name__ == "__main__":
    sample_input = """
    Internal freewheels TFS-12 inner diameter 12mm outer diameter 35mm width 13mm with keyway at inner ring and radial keyways at outer ring SKU: TFS12
    These sprag freewheels without own bearing support in premium quality can be used as indexing freewheels, backstops or overrunning clutches. They allow a compact design and are installed in housings provided by the customer. 
    """
    
    print("🚀 Starting Zero-Manual Intervention Pipeline...\n")
    final_state = video_pipeline.invoke({"raw_product_info": sample_input})
    
    print("\n[Final Output JSON from LLM]:")
    print(json.dumps(final_state['script_json'], indent=2))
    print("\nNext step: Run 'npx hyperframes render . \"threejs-split\" \"main_ad.mp4\"' in your terminal!")