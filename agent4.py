import json
import os
import re
import textwrap
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATE
# ═══════════════════════════════════════════════════════════════════════════════
class TemplateState(TypedDict):
    storyboard: str
    scenes: list
    layout_specs: list
    model_js: str
    animate_model_block: str
    current_code: str

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLM SETUP
# ═══════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

llm_json = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.0, 
    api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}},
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETERMINISTIC HELPERS  (Python Logic)
# ═══════════════════════════════════════════════════════════════════════════════
def placement_to_css(placement_str: str) -> str:
    p = (placement_str or "").lower()
    if any(x in p for x in ["top", "header"]):
        return "position:absolute;top:5%;left:50%;transform:translateX(-50%);width:90%;text-align:center;z-index:10;"
    if any(x in p for x in ["left"]):
        return "position:absolute;top:50%;left:5%;transform:translateY(-50%);width:42%;text-align:left;z-index:10;"
    if any(x in p for x in ["right"]):
        return "position:absolute;top:50%;right:5%;transform:translateY(-50%);width:42%;text-align:right;z-index:10;"
    if any(x in p for x in ["bottom"]):
        return "position:absolute;bottom:8%;left:50%;transform:translateX(-50%);text-align:center;width:80%;z-index:10;"
    if any(x in p for x in ["flanking", "2 bullet", "either side"]):
        return "position:absolute;top:55%;left:50%;transform:translate(-50%,-50%);width:85%;display:flex;flex-direction:row;justify-content:space-between;align-items:flex-start;gap:60px;z-index:10;"
    return "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;width:80%;z-index:10;"

def pt_to_px(pt: float | None, fallback: int = 36) -> int:
    return round(pt * 1.333) if pt else fallback

def build_model_js(model_windows: list) -> tuple[str, str]:
    if not model_windows:
        return ("let model = null;\nlet modelMaxDim = 1;", "// no model")

    vis_parts = [f"(t >= {w['start']} && t < {w['end']})" for w in model_windows]
    vis_condition = " || ".join(vis_parts)

    lines = [
        "const loader = new GLTFLoader();",
        "let model = null;",
        "let modelMaxDim = 1.0;",
        "const modelProxy = { rotY: 0, posY: 0, scaleF: 1.0 };",
        ""
    ]

    for w in model_windows:
        sm = w.get("scale_multiplier", 1.0)
        start, end = w["start"], w["end"]
        dur = end - start

        if w.get("drop"):
            lines.append(f"tl.set(modelProxy, {{ scaleF: {sm}, posY: 7, rotY: 0 }}, {start});")
            lines.append(f"tl.to(modelProxy, {{ posY: -0.4, duration: 2.2, ease: 'bounce.out', onUpdate: () => {{ if(model) model.position.y = modelProxy.posY; }} }}, {start});")
        else:
            lines.append(f"tl.set(modelProxy, {{ scaleF: {sm}, posY: 0, rotY: 0 }}, {start});")

        if w.get("spin"):
            lines.append(f"tl.to(modelProxy, {{ rotY: Math.PI * 2, duration: {dur}, ease: 'none', onUpdate: () => {{ if(model) model.rotation.y = modelProxy.rotY; }} }}, {start});")

    lines += [
        "loader.load('./product.glb', (gltf) => {",
        "  model = gltf.scene;",
        "  const box = new THREE.Box3().setFromObject(model);",
        "  const size = box.getSize(new THREE.Vector3());",
        "  modelMaxDim = Math.max(size.x, size.y, size.z) || 1.0;",
        "  model.traverse((child) => { if (child.isMesh) { child.geometry.center(); } });",
        "  model.visible = false;",
        "  threeScene.add(model);",
        "});"
    ]

    pos_x_lines = [f"if (t >= {w['start']} && t < {w['end']}) model.position.x = {w.get('pos_x', 0)};" for w in model_windows]
    
    animate_block = textwrap.dedent(f"""\
        if (model) {{
          const t = tl.time();
          model.visible = {vis_condition};
          if (model.visible) {{
            model.scale.setScalar((1.0 / modelMaxDim) * modelProxy.scaleF);
            model.rotation.y = modelProxy.rotY;
            model.position.y = modelProxy.posY;
            {chr(10).join(pos_x_lines)}
          }}
        }}""")

    return "\n".join(lines), animate_block

# ═══════════════════════════════════════════════════════════════════════════════
# 4. NODE 1 — PARSE STORYBOARD (LLM)
# ═══════════════════════════════════════════════════════════════════════════════
def parse_storyboard(state: TemplateState) -> TemplateState:
    print("🧠 [Node 1] Extracting Data via LLM ...")
    prompt = f"""
Parse the storyboard into a strict JSON object with a "scenes" array.
Return ONLY valid JSON. 

Schema for each scene:
{{
  "scene_number": <int>,
  "background_asset": <string|null>, 
  "background_type": <"image"|"video"|"none">,
  "title": {{
    "text": <string|null>,
    "placement": <string>, 
    "font_family": <string>, 
    "font_weight": <number>, 
    "font_size_pt": <number>,
    "color_hex": <string>, 
    "effects": [<string>] 
  }},
  "body_text": {{
    "content": <string|null>,
    "placement": <string>,
    "layout_hint": <string>, 
    "font_family": <string>,
    "font_weight": <number>,
    "font_size_pt": <number>,
    "color_hex": <string>,
    "effects": [<string>]
  }},
  "model_3d": {{
    "has_model": <boolean>,
    "scale_multiplier": <number>,
    "placement": <string>, 
    "animation": <string> 
  }}
}}

STORYBOARD:
{state["storyboard"]}
"""
    try:
        response = llm_json.invoke([HumanMessage(content=prompt)])
        scenes = json.loads(response.content).get("scenes", [])
        print(f"  → Parsed {len(scenes)} scenes successfully.")
        return {**state, "scenes": scenes}
    except Exception as e:
        print(f"  → Parsing Error: {e}")
        return {**state, "scenes": []}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. NODE 2 — COMPUTE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
def resolve_layout(state: TemplateState) -> TemplateState:
    print("📐 [Node 2] Calculating Spatial Math & JS Logic ...")
    scenes = state["scenes"]
    layout_specs, model_windows = [], []

    for i, s in enumerate(scenes):
        sn = s.get("scene_number", i + 1)
        start, end = (sn - 1) * 5, sn * 5
        
        t_data, b_data, m_data = s.get("title", {}), s.get("body_text", {}), s.get("model_3d", {})

        if m_data.get("has_model"):
            placement = (m_data.get("placement") or "").lower()
            anim = (m_data.get("animation") or "").lower()
            pos_x = 3.2 if "right" in placement else -3.2 if "left" in placement else 0.0
            
            model_windows.append({
                "scene_number": sn, "start": start, "end": end,
                "spin": any(x in anim for x in ["spin", "360"]),
                "drop": any(x in anim for x in ["drop", "fall"]),
                "scale_multiplier": m_data.get("scale_multiplier", 1.0),
                "pos_x": pos_x
            })

        layout_specs.append({
            "scene_number": sn, "start": start, "end": end,
            "bg_asset": s.get("background_asset"),
            "bg_type": s.get("background_type", "none"),
            "title_css": placement_to_css(t_data.get("placement", "")),
            "body_css": placement_to_css(b_data.get("placement", "") + " " + b_data.get("layout_hint", "")),
            "title_px": pt_to_px(t_data.get("font_size_pt")),
            "body_px": pt_to_px(b_data.get("font_size_pt"))
        })

    model_js, animate_block = build_model_js(model_windows)
    return {**state, "layout_specs": layout_specs, "model_js": model_js, "animate_model_block": animate_block}

# ═══════════════════════════════════════════════════════════════════════════════
# 6. NODE 3 — BUILD HTML
# ═══════════════════════════════════════════════════════════════════════════════
def generate_html_deterministic(state: TemplateState) -> TemplateState:
    print("⚡ [Node 3] Compiling DOM & CSS ...")
    scenes = state["scenes"]
    layout_specs = state["layout_specs"]
    total_dur = len(scenes) * 5

    dom_scenes = ""
    gsap_tweens = ""

    for ls, s in zip(layout_specs, scenes):
        sn = ls["scene_number"]
        t_data, b_data = s.get("title", {}), s.get("body_text", {})
        
        # ── Background Logic ──
        bg_html = ""
        is_vid = ls["bg_type"] == "video"
        if ls["bg_type"] == "image":
            bg_html = f'<img id="scene-{sn}-bg" class="bg-media" src="./{ls["bg_asset"]}">'
        elif is_vid:
            # FIX 1a: Removed autoplay. Video will be controlled strictly by GSAP.
            bg_html = f'<video id="scene-{sn}-bg" class="bg-media" muted loop playsinline><source src="./{ls["bg_asset"]}" type="video/mp4"></video>'

        # ── Text Logic ──
        title_html = ""
        if t_data.get("text"):
            effects_css = "text-shadow: 0 0 10px rgba(0,255,255,0.8);" if "pulse" in str(t_data.get("effects")).lower() else "text-shadow: 2px 4px 12px rgba(0,0,0,0.8);"
            anim_css = "animation: pulseGlow 2s infinite;" if "pulse" in str(t_data.get("effects")).lower() else ""
            t_style = f"{ls['title_css']} font-family: '{t_data.get('font_family', 'sans-serif')}'; font-weight: {t_data.get('font_weight', 400)}; font-size: {ls['title_px']}px; color: {t_data.get('color_hex', '#FFF')}; {effects_css} {anim_css}"
            title_html = f'<div id="scene-{sn}-title" style="{t_style}">{t_data.get("text")}</div>'

        body_html = ""
        if b_data.get("content"):
            content_str = b_data.get("content", "")
            content_str = content_str.replace("skies", '<span style="color:#0FF;text-shadow:0 0 8px #0FF">skies</span>')
            content_str = content_str.replace("right place", '<span style="color:#0FF;text-shadow:0 0 8px #0FF">right place</span>')

            anim_css = "animation: heartbeat 0.9s infinite;" if "heartbeat" in str(b_data.get("effects")).lower() else ""
            stroke_css = "-webkit-text-stroke: 3px #000; paint-order: stroke fill;" if "stroke" in str(b_data.get("effects")).lower() else ""
            
            b_style = f"{ls['body_css']} font-family: '{b_data.get('font_family', 'sans-serif')}'; font-weight: {b_data.get('font_weight', 400)}; font-size: {ls['body_px']}px; color: {b_data.get('color_hex', '#FFF')}; {anim_css} {stroke_css}"

            if "flanking" in (b_data.get("layout_hint") or "").lower():
                items = [i.strip() for i in content_str.split("|")]
                mid = len(items) // 2
                left = "".join([f'<div class="text-box">{i}</div>' for i in items[:mid]])
                right = "".join([f'<div class="text-box">{i}</div>' for i in items[mid:]])
                body_html = f'<div id="scene-{sn}-body" style="{b_style}"><div class="bullet-col">{left}</div><div class="bullet-col">{right}</div></div>'
            else:
                body_html = f'<div id="scene-{sn}-body" style="{b_style}">{content_str}</div>'

        dom_scenes += f'<div id="scene-{sn}" class="scene-container">\n  {bg_html}\n  <div class="scene-content">{title_html}\n{body_html}</div>\n</div>\n'
        
        # ── GSAP Logic ──
        # FIX 1b: Tie video playback strictly to the timeline scene window
        if is_vid:
            on_start = f"const v = document.getElementById('scene-{sn}-bg'); if(v) {{ v.currentTime = 0; v.play().catch(e=>console.log(e)); }}"
            on_complete = f"const v = document.getElementById('scene-{sn}-bg'); if(v) v.pause();"
            gsap_tweens += f"tl.fromTo('#scene-{sn}', {{opacity: 0}}, {{opacity: 1, duration: 0.4, onStart: () => {{ {on_start} }} }}, {ls['start']});\n"
            gsap_tweens += f"tl.to('#scene-{sn}', {{opacity: 0, duration: 0.4, onComplete: () => {{ {on_complete} }} }}, {ls['end'] - 0.4});\n"
        else:
            gsap_tweens += f"tl.fromTo('#scene-{sn}', {{opacity: 0}}, {{opacity: 1, duration: 0.4}}, {ls['start']});\n"
            gsap_tweens += f"tl.to('#scene-{sn}', {{opacity: 0, duration: 0.4}}, {ls['end'] - 0.4});\n"


    # FIX 2: Added RoomEnvironment for instant photorealistic lighting calculations
    master_html = f"""<!DOCTYPE html>
<html>
<head>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Roboto:wght@300;400&family=Open+Sans:wght@400;600&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin:0; background:#0a0a0a; overflow:hidden; color:white; }}
    #webgl-canvas {{ position:absolute;top:0;left:0;width:100%;height:100%;z-index:1; }}
    #ui-layer {{ position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;pointer-events:none; }}
    .scene-container {{ position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;overflow:hidden; }}
    .bg-media {{ position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;z-index:0; }}
    .scene-content {{ position:relative;z-index:2;width:100%;height:100%; }}
    .bullet-col {{ display:flex;flex-direction:column;align-items:flex-start;gap:10px;flex:1; }}
    .text-box {{ display:inline-block;padding:10px 24px;border-radius:6px;margin:6px 0; background:rgba(30,30,30,0.7); }}
    
    @keyframes pulseGlow {{
      0%, 100% {{ filter: brightness(1); }}
      50% {{ filter: brightness(1.5); text-shadow: 0 0 30px currentColor, 0 0 60px currentColor; }}
    }}
    @keyframes heartbeat {{
      0%, 100% {{ transform: scale(1); }}
      50% {{ transform: scale(1.1); }}
    }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js"></script>
</head>
<body>
  <div id="root" data-composition-id="main-composition" data-width="1920" data-height="1080" data-duration="{total_dur}">
    <canvas id="webgl-canvas"></canvas>
    <div id="ui-layer">
      {dom_scenes}
    </div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
    import {{ RoomEnvironment }} from 'three/addons/environments/RoomEnvironment.js';

    const canvas = document.querySelector('#webgl-canvas');
    const threeScene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1920/1080, 0.1, 100);
    camera.position.set(0, 0, 5);
    
    const renderer = new THREE.WebGLRenderer({{ canvas, alpha:true, antialias:true }});
    renderer.setSize(1920, 1080);
    
    // FIX 2b: Setup proper color spacing and tone mapping for PBR models
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // FIX 2c: Generate studio lighting environment map
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    threeScene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

    const tl = gsap.timeline({{ paused: true }});
    window.__timelines = {{ "main-composition": tl }};

    {gsap_tweens}
    {state['model_js']}

    function animate() {{
      requestAnimationFrame(animate);
      {state['animate_model_block']}
      renderer.render(threeScene, camera);
    }}
    animate();
    tl.play();
  </script>
</body>
</html>"""
    
    return {**state, "current_code": master_html}

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ROUTING & GRAPH COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════
def write_output(state: TemplateState) -> TemplateState:
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(state["current_code"])
    print("💾 [Node 4] index.html strictly compiled and saved.")
    return state

def build_agent() -> StateGraph:
    g = StateGraph(TemplateState)
    g.add_node("parse_storyboard", parse_storyboard)
    g.add_node("resolve_layout", resolve_layout)
    g.add_node("generate_html", generate_html_deterministic)
    g.add_node("write_output", write_output)

    g.set_entry_point("parse_storyboard")
    g.add_edge("parse_storyboard", "resolve_layout")
    g.add_edge("resolve_layout", "generate_html")
    g.add_edge("generate_html", "write_output")
    g.add_edge("write_output", END)
    
    return g.compile()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
STORYBOARD_OUTPUT = """
Scene 1
Background: image1.jpeg
Title: RMX Drone - 5600
Title Placement: Top of the page, centered
Title Design: Font: Montserrat Black (Sans-serif), Color: Crisp White (#FFFFFF), Effect: Subtle dark drop-shadow for readability, Size: 72pt, All Caps.
Text: Want to see the skies? You are at the right place.
Text Placement: Left side of the page, vertically centered
Text Design: Font: Roboto Light, Color: Silver/Light Gray (#E0E0E0), Effect: Glowing neon cyan (#00FFFF) outline on keywords "skies" and "right place", Size: 36pt.
3D Model: product.glb
Model Details: Size: 0.4x, spinning 360 degrees on the right side of the page.

Scene 2
Background: image2.jpeg
Title: Record in 4K
Title Placement: Top of the page, centered
Title Design: Font: Montserrat Bold, Color: Neon Cyan (#00FFFF), Effect: Pulsing glow animation, Size: 64pt.
Text: Carbon Frame | Light-weight | 10-hour battery | Available in 4 colors
Text Placement: 2 bullet points on the left, 2 bullet points on the right, flanking the center.
Text Design: Font: Open Sans Semi-Bold, Color: White (#FFFFFF), Effect: Encased in semi-transparent dark gray bounding boxes (Opacity 70%) to ensure text pops against image2.jpeg, Size: 28pt.

Scene 3
Background: bg_action.mp4
Text: BUY NOW
Text Placement: Bottom of the screen, centered
Text Design: Font: Arial Black, Color: High-visibility Orange (#FF5722), Effect: Heavy black stroke/outline, slight scale-up animation (heartbeat effect) to drive click-throughs, Size: 80pt.
3D Model: product.glb
Model Details: Size: 0.2x, drops from top to bottom, landing right above the "BUY NOW" text.
"""

if __name__ == "__main__":
    print("🚀 Running Agent (Deterministic Engine)...\n")
    agent = build_agent()
    agent.invoke({"storyboard": STORYBOARD_OUTPUT, "scenes": [], "layout_specs": [], "model_js": "", "animate_model_block": "", "current_code": ""})
    print("\n🏁 Process Complete.")