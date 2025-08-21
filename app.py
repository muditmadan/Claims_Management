import streamlit as st
import json
import tempfile
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import requests
import re
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    import streamlit as st
    st.error("PyPDF2 is not installed. Please install it with 'pip install PyPDF2' to enable PDF support.")
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors
from openai import AzureOpenAI

load_dotenv()

# Configure Azure OpenAI
azure_openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Page config
st.set_page_config(
    page_title="Car Damage Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    :root {
        --bg: #f8fafc;
        --surface: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
        --shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
        --shadow-hover: 0 6px 18px rgba(17, 24, 39, 0.10);
        --accent: #111827; /* subtle dark accent, no blues */
        --success: #16a34a;
        --warn: #f59e0b;
        --danger: #ef4444;
    }
    body, .stApp { background: var(--bg) !important; font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif; }
    .main-header {
        font-size: 3rem; font-weight: 800; color: var(--text); text-align: center;
        margin-bottom: 2rem; letter-spacing: .5px;
    }
    .section-header {
        font-size: 1.25rem; font-weight: 700; color: var(--text);
        margin: 1.75rem 0 .75rem; border-left: 4px solid var(--border);
        padding-left: 12px; background: var(--surface); border-radius: 6px; box-shadow: var(--shadow);
    }
    .result-box, .damage-item, .rejected-item, .missing-item, .additional-item {
        background: var(--surface); padding: .9rem 1rem; border-radius: 10px; margin: .6rem 0;
        border: 1px solid var(--border); box-shadow: var(--shadow); color: var(--text);
    }
    .damage-item { border-left: 4px solid var(--accent); }
    .damage-item:hover { box-shadow: var(--shadow-hover); }
    .missing-item { border-left: 4px solid var(--warn); }
    .additional-item { border-left: 4px solid var(--danger); }
    .rejected-item { border-left: 4px solid var(--danger); }
    .tooltip { position: relative; display: inline-block; cursor: pointer; }
    .tooltip .tooltiptext {
        visibility: hidden; min-width: 260px; background: var(--text); color: #fff; text-align: left;
        border-radius: 8px; padding: 10px 14px; position: absolute; z-index: 2; top: -5px; left: 110%;
        opacity: 0; transition: opacity .2s ease; font-size: .95em; box-shadow: var(--shadow);
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    .info-icon { color: var(--muted); margin-left: 8px; font-size: 1.1em; vertical-align: middle; }
    .stButton>button {
        background: var(--text); color: #fff; font-weight: 700; border: 1px solid var(--text);
        border-radius: 10px; padding: .65rem 1.2rem; font-size: 1rem; box-shadow: var(--shadow);
        transition: transform .04s ease, box-shadow .2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: var(--shadow-hover); }
    .stDownloadButton>button, .download-btn {
        display: inline-block; background: var(--success); color: #fff !important; font-weight: 700;
        border: 1px solid var(--success); border-radius: 10px; padding: .65rem 1.2rem; font-size: 1rem;
        box-shadow: var(--shadow); text-decoration: none;
    }
    .stDownloadButton>button:hover, .download-btn:hover { box-shadow: var(--shadow-hover); }
    .stExpanderHeader { font-size: 1.05rem !important; font-weight: 600 !important; color: var(--text) !important; }
    .stTextArea textarea {
        background: #fff !important; border-radius: 10px !important; font-size: 1.02rem !important;
        color: var(--text) !important; border: 1.5px solid var(--border) !important;
    }
    .stAlert { border-radius: 10px !important; font-size: 1.02rem !important; }
    .stImage>img { border-radius: 12px !important; box-shadow: var(--shadow); }
    /* Hover-to-reveal reason under AI-interpreted items */
    .reason-on-hover { display: none; font-size: .92em; color: var(--muted); margin-top: 6px; }
    .damage-item:hover .reason-on-hover, .rejected-item:hover .reason-on-hover { display: block; }
</style>
""", unsafe_allow_html=True)

# Helper: strip internal category prefixes from reasons for display
def _strip_reason_categories(reason: str) -> str:
    try:
        if not reason:
            return reason
        s = str(reason)
        # Remove our internal category sentences if present, case-insensitive
        s = re.sub(r"(?i)\bno visible damage in the damage report for this part\.\s*", "", s)
        s = re.sub(r"(?i)\bnot possible according to the claim story\.\s*", "", s)
        # Tidy leftover separators
        return s.strip().lstrip('-:; ').strip()
    except Exception:
        return reason

# Helper: render a direct PDF download link (no Streamlit rerun)
def render_pdf_download_link(pdf_bytes: bytes, filename: str, label: str = "📄 Download PDF Report"):
    try:
        if not pdf_bytes:
            st.info("PDF not ready yet.")
            return
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f"<a class='download-btn' href='data:application/pdf;base64,{b64}' download='{filename}'>{label}</a>"
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Unable to create download link: {e}")

def summarize_story(story, model_name="gpt-4o"):
    prompt = f"""
    Summarize the following accident story in 2-3 sentences, focusing on what happened and the main forces involved.
    STORY: {story}
    """
    response = azure_openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert insurance analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=200
    )
    content = response.choices[0].message.content
    if not content:
        return ""
    return content.strip()


def reason_about_impacts(summary, model_name="gpt-4o"):
    prompt = f"""
    Based on this summary, reason step-by-step about which vehicle parts are likely to be impacted, considering both direct and indirect effects. Explain your reasoning.
    SUMMARY: {summary}
    """
    response = azure_openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert in vehicle accident analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=300
    )
    content = response.choices[0].message.content
    if not content:
        return ""
    return content.strip()


def extract_explicit_and_implicit_parts(story, reasoning, model_name="gpt-4o"):
    prompt = f"""
    The following is a policy holder's story and a reasoning about the likely impacts:
    STORY: {story}
    REASONING: {reasoning}

    List all vehicle parts that could be damaged  along with the reason why they are included in the list, including both those explicitly mentioned in the story and those inferred from the reasoning. Return ONLY a JSON array of tuples of part names and their reasons, e.g. [["rear bumper","due to collision with the wall on rear side"], ["trunk","it is in the rightmost side so direct impact"], ["tail lights","very fragile and got impacted directly"]].
    """
    response = azure_openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are an expert in vehicle damage assessment."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=400
    )
    text = response.choices[0].message.content
    if not text:
        return []
    start_idx = text.find('[')
    end_idx = text.rfind(']') + 1
    if start_idx != -1 and end_idx != 0:
        json_str = text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except Exception:
            return []
    return []


def extract_damage_parts_with_reasoning(policy_story: str, model_name: str = "gpt-4o"):
    """
    Use a LangGraph-style multi-step reasoning agent to extract potential damaged parts and provide reasoning.
    Returns: dict with 'reasoning' and 'parts' keys.
    """
    summary = summarize_story(policy_story, model_name)
    reasoning = reason_about_impacts(summary, model_name)
    parts = extract_explicit_and_implicit_parts(policy_story, reasoning, model_name)
    return {
        "summary": summary,
        "reasoning": reasoning,
        "parts": parts
    }

def extract_damage_parts_from_story(policy_story: str, model_name: str = "gpt-4o") -> list:
    """Extract potential damage parts from policy holder's story (single-step, legacy)"""
    result = extract_damage_parts_with_reasoning(policy_story, model_name)
    print(result)
    return result['parts'] if 'parts' in result else []

def generate_story_vs_visual_consistency_summary(claim_story: str, story_detected_parts: list, ai_detected_parts: list, model_name: str = "gpt-4o") -> str:
    """
    Generate a consistency analysis summary comparing claim story with AI-detected visual damages.
    Only considers claim story and AI-identified damaged parts from images.
    """
    prompt = f"""
    You are an expert vehicle accident analyst. Your task is to compare the `POTENTIAL PARTS FROM STORY` with the `DETECTED PARTS ARRAY` (from images) and generate a 3-4 line consistency analysis. 
    
    Evaluate if the visually detected damages logically support the accident description. For example, if the story claims impacts to the front, back, and left, but the detected parts only show front damage, your summary must explicitly state that the back and left impacts are not visually substantiated by the provided images. Focus only on the physical consistency between the story and the visual evidence.

    CLAIM STORY: {claim_story}

    POTENTIAL PARTS FROM STORY: {story_detected_parts}

    DETECTED PARTS ARRAY (from images): {ai_detected_parts}

    Instructions:
    - Compare what the accident story describes (impact directions, collision type) with what damage is actually visible in the images
    - Identify any discrepancies between expected damage locations and actual visible damage
    - Point out if claimed impact areas are not supported by visual evidence
    - Do NOT mention estimate copy, financial details, or repair costs
    - Focus exclusively on physical consistency between story description and visual damage evidence
    - Provide a clear, concise 3-4 line analysis

    Return only the consistency analysis text, no JSON format needed.
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert vehicle accident analyst specializing in consistency analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        if not content:
            return "Unable to generate consistency analysis."
        
        return content.strip()
        
    except Exception as e:
        print(f"Error generating consistency summary: {e}")
        return f"Error generating consistency analysis: {str(e)}"

def analyze_single_accident_consistency_with_vision(car_images: dict, claim_story: str, ai_detected_parts: list = None, model_name: str = "gpt-4o") -> str:
    """
    Enhanced visual damage analysis that:
    1. Identifies collision type from claim story
    2. Analyzes only relevant images based on collision type
    3. Uses AI-detected parts for comparison
    4. Uses multiple LLM calls for thorough analysis
    """
    try:
        # Basic validation
        if not car_images:
            return "ERROR: No car images provided for analysis"
        
        if not claim_story or not claim_story.strip():
            return "ERROR: No claim story provided for analysis"
        
        # Check if Azure OpenAI client is available
        if not azure_openai_client:
            return "ERROR: Azure OpenAI client not configured"
        
        # Step 1: Identify collision type by matching AI-interpreted parts and claim story
        collision_analysis_prompt = f"""
        Analyze the claim story and AI-detected damaged parts to determine the collision type and relevant views for analysis:

        CLAIM STORY: {claim_story}
        AI-DETECTED DAMAGED PARTS: {ai_detected_parts if ai_detected_parts else 'None detected'}

        Match the claim story with the damaged parts to determine:
        1. Primary collision type based on BOTH story and damaged parts location
        2. Which car view(s) would show the most relevant damage

        COLLISION TYPE LOGIC:
        - FRONT: If story mentions front impact AND parts include front bumper, headlights, grille, hood
        - REAR: If story mentions rear impact AND parts include rear bumper, taillights, trunk
        - SIDE: If story mentions side impact AND parts include doors, side panels, mirrors
        - MULTIPLE: If story describes multiple impacts OR parts span multiple areas (front+rear, front+side, etc.)

        Respond with only: "[COLLISION_TYPE]: [RELEVANT_VIEWS]"
        Examples:
        "FRONT: front"
        "REAR: rear" 
        "SIDE: left,right"
        "MULTIPLE: front,rear"
        """
        
        print(f"DEBUG: Analyzing collision type with AI parts: {ai_detected_parts}")
        
        collision_response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert accident analyst."},
                {"role": "user", "content": collision_analysis_prompt}
            ],
            temperature=0,
            max_tokens=75
        )
        
        collision_analysis = collision_response.choices[0].message.content.strip()
        print(f"DEBUG: Collision analysis result: {collision_analysis}")
        
        # Parse collision type and relevant views with better cleaning
        if ":" in collision_analysis:
            collision_type, relevant_views_str = collision_analysis.split(":", 1)
            collision_type = collision_type.strip()
            
            # Clean up the views string and split by comma
            relevant_views_str = relevant_views_str.strip()
            # Remove any quotes, brackets, or extra formatting
            relevant_views_str = relevant_views_str.replace('"', '').replace("'", "").replace('[', '').replace(']', '')
            relevant_views = [view.strip() for view in relevant_views_str.split(",") if view.strip()]
        else:
            # Fallback - analyze all available images
            relevant_views = list(car_images.keys())
            collision_type = "UNKNOWN"
        
        print(f"DEBUG: Determined collision type: {collision_type}, Relevant views: {relevant_views}")
        
        # Step 2: Select and prepare only relevant images
        relevant_image_data = []
        available_views = []
        
        print(f"DEBUG: Relevant views to analyze: {relevant_views}")
        print(f"DEBUG: Available car images: {list(car_images.keys())}")
        
        for view in relevant_views:
            if view in car_images:
                try:
                    image = car_images[view]
                    print(f"DEBUG: Processing image for view: {view}")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        image.save(tmp_file.name, format="PNG")
                        with open(tmp_file.name, "rb") as f:
                            img_bytes = f.read()
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        relevant_image_data.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                        available_views.append(view)
                        os.unlink(tmp_file.name)  # Clean up temp file
                        print(f"DEBUG: Successfully processed image for view: {view}")
                except Exception as img_error:
                    print(f"DEBUG: Error processing image for view {view}: {img_error}")
                    continue
        
        if not relevant_image_data:
            return f"UNABLE: No relevant images available for analysis. Needed: {relevant_views}, Available: {list(car_images.keys())}"
        
        # Step 3: Enhanced visual analysis with AI-detected parts context
        ai_parts_context = ""
        if ai_detected_parts:
            ai_parts_context = f"\n\nAI-DETECTED DAMAGED PARTS: {', '.join(ai_detected_parts)}"
        
        enhanced_prompt = f"""
        You are an expert vehicle accident analyst. Focus ONLY on detecting "mirror damage" patterns that indicate multiple separate impacts rather than single collisions.

        CLAIM STORY: {claim_story}
        COLLISION TYPE: {collision_type}
        ANALYZING VIEWS: {', '.join(available_views)}{ai_parts_context}

        CRITICAL ANALYSIS - Focus on this specific pattern:

        FOR {collision_type} COLLISION, look specifically for:
        - FRONT: Both LEFT and RIGHT headlights damaged BUT center grille/bumper is intact/undamaged
        - REAR: Both LEFT and RIGHT taillights damaged BUT center rear bumper is intact/undamaged  
        - SIDE: Multiple side panels damaged with gaps of undamaged areas between them

        INCONSISTENT PATTERN (mark as INCONSISTENT):
        - If you see damage on BOTH left and right sides while the connecting CENTER part is undamaged
        - This indicates two separate corner impacts instead of one central collision

        CONSISTENT PATTERN (mark as CONSISTENT):
        - Damage follows logical force transfer from a single impact point
        - OR damage is only on one side
        - OR center part is damaged along with sides (normal single impact pattern)
        - OR any other pattern that doesn't show the specific "mirror damage"

        Response Format:
        - If CONSISTENT (normal single collision pattern): Respond with only "CONSISTENT"
        - If INCONSISTENT (mirror damage pattern detected): Respond with "INCONSISTENT: [brief explanation of why]"

        Examples:
        - "CONSISTENT"
        - "INCONSISTENT: Both headlights damaged but center grille intact indicates separate corner impacts"
        """
        
        # Step 4: Perform visual analysis
        messages = [
            {"role": "system", "content": "You are an expert vehicle damage pattern analyst."},
            {
                "role": "user",
                "content": [{"type": "text", "text": enhanced_prompt}] + relevant_image_data
            }
        ]
        
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        if not result:
            return "UNKNOWN: Unable to analyze damage patterns"
        
        return result.strip()
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error analyzing single accident consistency: {error_msg}")
        
        # More specific error messages for debugging
        if "vision" in error_msg.lower():
            return "ERROR: Vision model not available or not configured properly"
        elif "token" in error_msg.lower() or "quota" in error_msg.lower():
            return "ERROR: Token limit or quota exceeded"
        elif "image" in error_msg.lower():
            return "ERROR: Image processing failed"
        elif "deployment" in error_msg.lower():
            return "ERROR: Azure OpenAI deployment issue"
        else:
            return f"ERROR: {error_msg[:100]}..."  # Show first 100 chars of error

def extract_parts_from_estimate_copy(estimate_text: str, model_name: str = "gpt-4o") -> list:
    """Extract car parts names from estimate copy text"""
    if not estimate_text or estimate_text.strip() == "":
        return []
    
    prompt = f"""
    You are an expert automotive repair estimate analyzer. Analyze the following repair estimate document and extract ALL vehicle part names mentioned in it.
    
    Look for parts listed under sections like: 'Parts', 'Parts & Labor', 'Replace', 'Repair', 'Body Parts', 'Components', 'Items', etc.
    Focus on identifying actual vehicle components that need repair or replacement.
    
    ESTIMATE COPY TEXT:
    {estimate_text}
    
    Extract only the actual vehicle part names from the estimate, not prices, labor costs, shop supplies, or other non-part items.
    If you find multiple entries for the same part, list it only once.
    Return ONLY a JSON array of part names, e.g. ["front bumper", "headlights", "windshield", "radiator", "door panel"]
    If no vehicle parts are found, return an empty array [].
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert automotive repair estimate analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        text = response.choices[0].message.content
        if not text:
            return []
        
        # Extract JSON array from response
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            try:
                parts_list = json.loads(json_str)
                # Ensure it's a list of strings
                if isinstance(parts_list, list):
                    return [str(part).strip() for part in parts_list if part]
                return []
            except Exception:
                return []
        return []
        
    except Exception as e:
        print(f"Error extracting parts from estimate: {e}")
        return []

def filter_damage_report(damage_report_text: str, potential_parts: list, model_name: str = "gpt-4o", estimate_text: str = "", estimate_parts: list = None, ai_detected_parts: list | None = None, ai_part_reasons: dict | None = None, claim_story_text: str = "") -> dict:
    """Filter damage report to only show damages related to potential parts from story"""
    
    # Use provided estimate_parts or fallback to empty list
    if estimate_parts is None:
        estimate_parts = []
    
    # Derive simple names array for potential parts (story)
    try:
        potential_part_names = [ (p[0] if isinstance(p, (list, tuple)) else p) for p in (potential_parts or []) ]
    except Exception:
        potential_part_names = []

    # Build a compact part->reason mapping to help the model write grounded reasons
    try:
        ai_part_reasons = ai_part_reasons or {}
        if not ai_part_reasons:
            # Derive from potential_parts if it contains [part, reason] tuples
            tmp_map = {}
            for p in (potential_parts or []):
                try:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        part_name = str(p[0]).strip()
                        part_reason = str(p[1]).strip()
                        if part_name and part_reason and part_name not in tmp_map:
                            tmp_map[part_name] = part_reason
                except Exception:
                    continue
            ai_part_reasons = tmp_map
    except Exception:
        pass

    prompt = f"""
    You are an expert insurance claims agent. You have received 4 car images (front, back, left, right) and a claim form story describing the accident. Your job is to analyze the detected damaged parts from all images and:

    1. Identify those parts that are likely to be damaged as a result of the same accident described in the claim form story. These should be listed as 'relevant_damages'. Always pick parts from the DETECTED PARTS ARRAY (AI-interpreted list); do not invent or rename parts.
    2. Flag any other parts that are damaged but you are doubtful about their connection to the described accident, but which could plausibly be affected due to a collision. These should be listed as 'doubtful_damages'.
    3. Remove and ignore any parts that are impossible to be affected by the described accident scenario, based on physical plausibility. For example, if the story says the car was hit from the front, then rear bumper damage is impossible and must be excluded, with a reason.
    4. For each part, provide a clear and concise 'reason' for its classification. For 'relevant_damages' specifically, write a concrete, specific one-liner grounded in: (a) evidence in the damage report/images (mention view or visible pattern like dents, cracks, misalignment), (b) the claim story impact direction/physics, and (c) where helpful, briefly reuse or adapt the AI-interpreted reason; avoid generic phrases.
    5. For rejected parts analysis, analyze ONLY the parts from the extracted estimate parts list below and reject them if they meet either criteria:
        a) Parts that show NO damage according to the damage report images from any view (front, back, left, right)
        b) Parts that might show damage in the images but CANNOT be plausibly linked to this specific accident based on the claim story (e.g., damage that appears to be from a previous accident, different impact direction, or unrelated incident)
        IMPORTANT: Only consider parts from the extracted estimate parts list. Do not include any parts from the claim story or generate any parts yourself. Use only the parts provided in the list below.
    6. For doubtful damages, analyze internal/hidden parts that could be damaged based on the accident scenario but are not visible in the images. Also, if an estimate part is present/mentioned but you are NOT ≥ 80% confident to either match or reject it, put that estimate part under 'doubtful_damages' with a brief reason (e.g., inspection required). Select 6-7 relevant internal/hidden parts from this comprehensive list based on collision type:
        • Front impact parts: Radiator, condenser, engine mounts, cooling fan, intercooler, timing components, front crossmember, transmission cooler, wiring harness, air intake system
        • Rear impact parts: Rear panel, boot floor, spare wheel housing, rear impact bar, rear crash sensors, fuel tank, wiring harness, rear suspension links, trunk latch mechanism
        • Side impact parts: Door beams, curtain airbags, seatbelt pretensioners, wiring inside doors, window regulator, power window motor, side impact sensors, B-pillar, seats and frames
        • Hidden/Internal damage: Frame rails, subframe, suspension mounts, steering column (front), rear axle beam (rear), floor pan, insulation, soundproofing materials
    7. For the summary section, your task is to compare the `POTENTIAL PARTS FROM STORY` ({potential_parts}) with the `DETECTED PARTS ARRAY` ({ai_detected_parts}) (from images) and generate a 3-4 line consistency analysis. Evaluate if the visually detected damages logically support the accident description. For example, if the story claims impacts to the front, back, and left, but the detected parts only show front damage, your summary must explicitly state that the back and left impacts are not visually substantiated by the provided images. Do not mention the estimate copy or financial details in this summary; focus only on the physical consistency between the story and the visual evidence.

    POTENTIAL PARTS FROM STORY (detailed): {potential_parts}
    POTENTIAL PART NAMES (story filter): {potential_part_names}
    DETECTED DAMAGED PARTS FROM IMAGES:
    {damage_report_text}
    DETECTED PARTS ARRAY (authoritative list; use exact strings for 'part' fields): {ai_detected_parts}
    AI-INTERPRETED PART REASONS (reference; do not copy verbatim, adapt and ground in evidence): {ai_part_reasons}
    EXTRACTED PARTS FROM ESTIMATE COPY: {estimate_parts}
    ESTIMATE COPY: {estimate_text}

    IMPORTANT FOR REJECTED PARTS: Use only the parts from the "EXTRACTED PARTS FROM ESTIMATE COPY" list above. Evaluate each estimate part STRICTLY in this order and include it in 'rejected_parts' ONLY when you are highly confident (≥ 80% confidence) about the rejection:
        1) PRESENCE CHECK FIRST: If the part is NOT mentioned/visible in the damage report text, then it MUST be rejected with reason "No visible damage in the damage report for this part" (optionally add a brief clause tied to the part/views). Do NOT use claim-story reasoning in this case.
        2) PHYSICS/STORY CHECK SECOND: Only if the part IS clearly present in the damage report text, assess consistency with the claim story and impact physics. If it cannot plausibly be linked to this specific accident, then reject it with reason "Not possible according to the claim story" (optionally add a brief clause referencing impact direction/physics).
        NEVER use "Not possible according to the claim story" for a part that is absent from the damage report.
    If your confidence is not high for rejecting a part, do NOT include it under 'rejected_parts'. Instead, include that estimate part under 'doubtful_damages' with a concise reason (e.g., linkage to this accident is uncertain; needs inspection). Do not leave the part out.
    REASONING RULES FOR 'rejected_parts': Choose EXACTLY ONE of the two templates above and produce a single, concise sentence starting with that template, followed by a brief specific clause (mention views or mismatch with front/rear/side impact as appropriate). If BOTH conditions could apply, PREFER the presence outcome and output ONLY the "No visible damage in the damage report for this part" reason (do not add the claim-story reason). Never include both templates in the same reason. Avoid hedgy terms like maybe/might/possibly.

    COVERAGE MANDATE FOR ESTIMATE PARTS:
    - EVERY part from the EXTRACTED PARTS FROM ESTIMATE COPY must appear in EXACTLY ONE of these lists: 'relevant_damages', 'doubtful_damages', or 'rejected_parts'.
    - All estimate copy parts must be categorized: each part should be placed in either matched (relevant_damages), doubtful, or rejected categories based on the following logic: (1) matched if the part shows visible damage in images AND aligns with the claim story impact direction, (2) rejected if the part shows no visible damage in images OR contradicts the claim story physics, (3) doubtful if the part could be damaged but requires inspection OR if confidence is not ≥80% for either matching or rejecting.
    - Do NOT omit or duplicate any estimate part. If uncertain between match and reject, choose 'doubtful_damages' and provide a clear one-line reason.
    - When placing an estimate part into 'relevant_damages', use the closest matching part name from the DETECTED PARTS ARRAY (authoritative list). If there is no sensible match in the DETECTED PARTS ARRAY, do not force it into 'relevant_damages'; use 'doubtful_damages' or 'rejected_parts' per rules above.

    MATCHING GUIDELINES FOR 'relevant_damages':
    - Always choose parts from the DETECTED PARTS ARRAY above. Do not generate or rename parts.
    - Include a part when it is likely to be damaged given the claim story and impact direction. Alignment with POTENTIAL PART NAMES from the story is helpful but not mandatory word-for-word.
    - CRITICAL: Only show parts in 'relevant_damages' that appear in BOTH the estimate copy list ({estimate_parts}) AND the AI-interpreted potentially impacted parts list ({ai_detected_parts}). Parts must exist in both lists to be considered for matching. Note: Allow for slight name variations that mean the same thing (e.g., "front bumper" vs "bumper front", "headlight" vs "head light", "door panel" vs "door", "fender" vs "front fender", "mirror" vs "side mirror", etc.). Consider semantic similarity, not just exact string matching.
    - If there is no overlap between likely parts and the DETECTED PARTS ARRAY, leave 'relevant_damages' empty.

    Return your analysis in this JSON format:
    {{
        "relevant_damages": [
            {{
                "part": "front bumper",
                "damage_description": "Front bumper dented and scratched",
                "severity": "medium",
                "found_in_report": true,
                "reason": "Front-view photo shows dents and paint transfer on the bumper consistent with a head-on impact; claim story reports a front collision, aligning with this damage."
            }}
        ],
        "doubtful_damages": [
            {{
                "part": "side panel",
                "damage_description": "Minor scratch on passenger side",
                "severity": "low",
                "reason": "Could be affected by collision but not clearly related to the described accident."
            }},
            {{
                "part": "radiator",
                "damage_description": "Internal cooling system component",
                "severity": "medium",
                "reason": "Front impact forces could damage radiator but not visible in images. Requires inspection."
            }},
            {{
                "part": "engine mounts",
                "damage_description": "Engine mounting system",
                "severity": "high",
                "reason": "Impact forces can cause engine mount failure, affecting engine stability and alignment."
            }},
            {{
                "part": "front crossmember",
                "damage_description": "Structural front component",
                "severity": "high",
                "reason": "Front collision can damage crossmember, affecting structural integrity."
            }},
            {{
                "part": "door beams",
                "damage_description": "Side impact protection",
                "severity": "medium",
                "reason": "Side impact can damage internal door reinforcement beams."
            }},
            {{
                "part": "frame rails",
                "damage_description": "Main structural component",
                "severity": "high",
                "reason": "Impact forces can damage frame rails, affecting vehicle structural integrity."
            }},
            {{
                "part": "curtain airbags",
                "damage_description": "Side safety system",
                "severity": "high",
                "reason": "Side impact can damage curtain airbag system, affecting passenger safety."
            }}
        ],
        "rejected_parts": [
            {{
                "part": "headlights",
                "reason": "Listed in estimate copy but no damage found in the damage report images. No visible damage detected in any view."
            }},
            {{
                "part": "rear quarter panel",
                "reason": "Listed in estimate copy and damage visible in images, but damage pattern suggests previous accident unrelated to front collision described in claim story."
            }}
        ],
        "summary": "Brief summary of findings",
        "recommendation": "Approve/Investigate/Deny"
    }}

    
    STRICT RULES:
    - 'relevant_damages' MUST be selected only from the DETECTED PARTS ARRAY (AI-interpreted parts). Do not output parts not present in that list.
    - Include parts in 'relevant_damages' when you assess they are likely to be damaged due to the same accident as described in the claim form story, with soft alignment to POTENTIAL PART NAMES.
    - For 'doubtful_damages', select 6-7 relevant internal/hidden parts from the provided lists based on collision type (front/rear/side impact). Choose parts that could plausibly be affected by the impact forces but are not visible in images.
    - Do NOT include any parts that are impossible to be affected by the described accident scenario (e.g., rear impact parts in a front collision). Always provide a reason for exclusion.
    - For each part, provide a clear 'reason' for its inclusion or exclusion, referencing the physical plausibility and accident details.
    - Be strict and accurate in your analysis. Never include impossible damages.
    - 'rejected_parts' should ONLY include parts from the "EXTRACTED PARTS FROM ESTIMATE COPY" list that meet either criteria: (a) NOT found in the damage report text (presence check fails) — use the reason starting with "No visible damage in the damage report for this part", OR (b) are present in the damage report text but cannot be plausibly linked to this specific accident based on the claim story — use the reason starting with "Not possible according to the claim story". Include a part in 'rejected_parts' ONLY when you are highly certain. Use only the parts provided in the extracted list. Do not include any parts from claim story or generate parts yourself.
    - For doubtful damages, focus on selecting the most relevant internal components from the provided lists that match the collision type and impact direction described in the claim story.
    - MANDATORY COVERAGE: Every single estimate part from {estimate_parts} must be classified into exactly one of 'relevant_damages', 'doubtful_damages', or 'rejected_parts'. NO EXCEPTIONS. If unsure, prefer 'doubtful_damages' with a succinct reason (e.g., inspection required). Never omit an estimate part from these three lists. Double-check that each estimate part appears exactly once across all three categories before finalizing your response.
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an insurance claims analyst. Filter damage reports based on claimed parts."
                },
                {
                    "role": "user",  
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        if not response_text:
            return {
                "relevant_damages": [],
                "rejected_parts": [],
                "additional_damages": [],
                "summary": "No response received",
                "recommendation": "Manual review needed"
            }
            
        # Robust JSON extraction: get only the first valid JSON object
        import json
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(json_str)
                
                # Ensure recommendation is not empty
                if not obj.get('recommendation') or obj.get('recommendation').strip() == "":
                    # Determine recommendation based on rejected parts and consistency
                    rejected_count = len(obj.get('rejected_parts', []))
                    doubtful_count = len(obj.get('doubtful_damages', []))
                    
                    if rejected_count > 2:
                        obj['recommendation'] = "Investigate"
                    elif rejected_count > 0 or doubtful_count > 3:
                        obj['recommendation'] = "Investigate"
                    else:
                        obj['recommendation'] = "Approve"
                
                # Generate specialized summary using the new function
                # Extract simple part names from potential_parts for the summary function
                story_parts = []
                try:
                    for p in (potential_parts or []):
                        if isinstance(p, (list, tuple)) and len(p) >= 1:
                            story_parts.append(p[0])
                        else:
                            story_parts.append(str(p))
                except Exception:
                    story_parts = potential_parts or []
                
                # Generate the specialized summary
                specialized_summary = generate_story_vs_visual_consistency_summary(
                    claim_story=claim_story_text or "No claim story provided",
                    story_detected_parts=story_parts,
                    ai_detected_parts=ai_detected_parts or [],
                    model_name=model_name
                )
                
                # Replace the summary with our specialized one
                obj['summary'] = specialized_summary
                
                return obj
            except Exception as e:
                return {
                    "relevant_damages": [],
                    "rejected_parts": [],
                    "additional_damages": [],
                    "summary": f"JSON parse error: {str(e)}",
                    "recommendation": "Manual review needed"
                }
        else:
            return {
                "relevant_damages": [],
                "rejected_parts": [],
                "additional_damages": [],
                "summary": "Failed to parse response",
                "recommendation": "Manual review needed"
            }
            
    except Exception as e:
        st.error(f"Error filtering report: {e}")
        return {
            "relevant_damages": [],
            "rejected_parts": [],
            "additional_damages": [],
            "summary": f"API error: {str(e)}",
            "recommendation": "Technical error"
        }

def analyze_image_with_gpt_vision(image: Image.Image, model_name: str = "gpt-4o") -> list:
    """
    Analyze the uploaded image using Azure OpenAI GPT-4 Vision model to extract damaged vehicle parts.
    Returns a list of tuples: [(part, reason), ...]
    """
    # Convert image to bytes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file.name, format="PNG")
        tmp_file.seek(0)
        img_bytes = tmp_file.read()
    prompt = (
        "You are an expert vehicle damage assessor. "
        "Analyze the uploaded car image and list all visible damaged parts. "
        "Focus on these specific vehicle components based on the image view: "
        "Front: Front bumper, grille, hood, headlights, fog lights, front fenders, windshield, side mirrors, DRLs, air intakes, front license plate. "
        "Rear: Rear bumper, trunk, tail lights, rear windshield, rear fenders, exhaust pipes, reverse lights, reflectors, spoiler, rear emblem, rear camera. "
        "Side: Doors, windows, door handles, side skirts, wheel arches, wheels, tires, fuel cap, turn signals, pillars (A, B, C). "
        "Visible extras: Roof rails, antenna, tow hook cover, high-mounted brake light, quarter glass. "
        "For each damaged part you identify, provide a short reason for why you think it is damaged. "
        "Be thorough and detailed in your analysis, checking each component carefully. "
        "Return ONLY a JSON array of tuples: [[\"part\", \"reason\"], ...]. "
        "If no damage is visible, return an empty array."
    )
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert vehicle damage assessor."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")}}
                    ]
                }
            ],
            max_tokens=400
        )
        text = response.choices[0].message.content
        # Try to extract JSON array from the response
        if not text:
            return []
        import json
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except Exception:
                return []
        return []
    except Exception as e:
        import streamlit as st
        st.error(f"Error analyzing image with GPT-4 Vision: {e}")
        return []

def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file object."""
    if PyPDF2 is None:
        return "[ERROR: PyPDF2 not installed]"
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_cause_of_loss(text: str, model_name: str = "gpt-4o") -> str:
    """Extract the cause of loss from the given text using LLM."""
    prompt = f"""
    Extract the main 'cause of loss' (i.e., the main reason for the insurance claim) from the following document. Return only a concise sentence or phrase describing the cause of loss. If not found, return 'Not found'.
    DOCUMENT: {text}
    """
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert insurance analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        content = response.choices[0].message.content
        if not content:
            return "Not found"
        return content.strip()
    except Exception:
        return "Not found"

def generate_matching_parts_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=None, rejected_parts=None, ai_interpreted_parts=None, consistency_report=None, view_mode: str = "without_dr"):
    # Use Streamlit cache to avoid regenerating PDF on rerun
    import streamlit as st
    @st.cache_data(show_spinner=False)
    def _generate_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=None, rejected_parts=None, ai_interpreted_parts=None, consistency_report=None, view_mode: str = "without_dr"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        title_style = styles['Title']
        normal_style = styles['Normal']
        heading_style = styles['Heading2']
        
        elements.append(Paragraph("Car Damage Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(f"<b>Damage Report File:</b> {damage_report_file}", normal_style))
        elements.append(Paragraph(f"<b>Model Used:</b> {model_used}", normal_style))
        elements.append(Spacer(1, 0.15*inch))
        elements.append(Paragraph(f"<b>Summary:</b> {summary}", normal_style))
        elements.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # AI-Interpreted Potentially Impacted Parts section
        if ai_interpreted_parts:
            elements.append(Paragraph("<b>AI-Interpreted Potentially Impacted Parts</b>", heading_style))
            for view, parts in ai_interpreted_parts.items():
                if parts:
                    elements.append(Paragraph(f"<b>{view.title()} View:</b>", normal_style))
                    for part_tuple in parts:
                        part = part_tuple[0] if isinstance(part_tuple, (list, tuple)) else part_tuple
                        reason = part_tuple[1] if isinstance(part_tuple, (list, tuple)) and len(part_tuple) > 1 else ""
                        elements.append(Paragraph(f"• {part}", normal_style))
                        if reason:
                            elements.append(Paragraph(f"  Reason: {reason}", normal_style))
                    elements.append(Spacer(1, 0.1*inch))
            elements.append(Spacer(1, 0.15*inch))
        
        # Matched Damages section
        elements.append(Paragraph("<b>Matched Damages</b>", heading_style))
        if not matching_parts:
            elements.append(Paragraph("No matching parts found.", normal_style))
        else:
            for damage in matching_parts:
                part = damage.get('part', '')
                desc = damage.get('damage_description', '')
                reason = damage.get('reason', '')
                elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
                if view_mode != "with_dr":
                    if desc:
                        elements.append(Paragraph(f"<b>Description:</b> {desc}", normal_style))
                    if reason:
                        elements.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
                elements.append(Spacer(1, 0.12*inch))
        
        # Doubtful Damages section
        if doubtful_parts:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Doubtful/Flagged Damaged Parts</b>", heading_style))
            for damage in doubtful_parts:
                part = damage.get('part', '')
                desc = damage.get('damage_description', '')
                reason = damage.get('reason', '')
                elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
                if desc:
                    elements.append(Paragraph(f"<b>Description:</b> {desc}", normal_style))
                # With Damage Report: omit reason; Without: include
                if view_mode != "with_dr" and reason:
                    elements.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
                elements.append(Spacer(1, 0.12*inch))
        
        # Rejected Parts section
        if rejected_parts:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Rejected Parts (Claimed but Not Found in Damage Report)</b>", heading_style))
            for rejected in rejected_parts:
                part = rejected.get('part', '')
                elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
                elements.append(Spacer(1, 0.12*inch))
        
        # Consistency Check section
        if consistency_report and view_mode != "with_dr":
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Accident Consistency Analysis</b>", heading_style))
            elements.append(Paragraph(consistency_report, normal_style))
        
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf
    return _generate_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=doubtful_parts, rejected_parts=rejected_parts, ai_interpreted_parts=ai_interpreted_parts, consistency_report=consistency_report, view_mode=view_mode)

def extract_claim_form_details(form_text: str, model_name: str = "gpt-4o") -> dict:
    """Extract key details from a claim form using LLM"""
    prompt = f"""
    Extract the following key details from this claim form document:
    
    DOCUMENT: {form_text}
    
    Return the analysis in this JSON format:
    {{
        "claimant_info": {{
            "name": "Full name of claimant",
            "phone": "Phone number",
            "email": "Email address",
            "address": "Full address"
        }},
        "vehicle_info": {{
            "make": "Vehicle make",
            "model": "Vehicle model",
            "year": "Vehicle year",
            "vin": "VIN number",
            "license_plate": "License plate number"
        }},
        "incident_info": {{
            "date": "Date of incident",
            "time": "Time of incident",
            "location": "Location of incident",
            "description": "Brief description of what happened"
        }},
        "insurance_info": {{
            "policy_number": "Insurance policy number",
            "coverage_type": "Type of coverage",
            "deductible": "Deductible amount"
        }},
        "damage_details": {{
            "damaged_parts": ["List of damaged parts"],
            "estimated_cost": "Estimated repair cost",
            "severity": "Overall damage severity (high/medium/low)"
        }},
        "summary": "Brief summary of the claim",
        "extraction_confidence": "High/Medium/Low based on completeness of information"
    }}
    
    If any information is not found in the document, use "Not found" as the value.
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert insurance claims processor. Extract structured information from claim forms."
                },
                {
                    "role": "user",  
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        if not response_text:
            return {
                "claimant_info": {"name": "Not found", "phone": "Not found", "email": "Not found", "address": "Not found"},
                "vehicle_info": {"make": "Not found", "model": "Not found", "year": "Not found", "vin": "Not found", "license_plate": "Not found"},
                "incident_info": {"date": "Not found", "time": "Not found", "location": "Not found", "description": "Not found"},
                "insurance_info": {"policy_number": "Not found", "coverage_type": "Not found", "deductible": "Not found"},
                "damage_details": {"damaged_parts": [], "estimated_cost": "Not found", "severity": "Not found"},
                "summary": "No response received",
                "extraction_confidence": "Low"
            }
            
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {
                "claimant_info": {"name": "Not found", "phone": "Not found", "email": "Not found", "address": "Not found"},
                "vehicle_info": {"make": "Not found", "model": "Not found", "year": "Not found", "vin": "Not found", "license_plate": "Not found"},
                "incident_info": {"date": "Not found", "time": "Not found", "location": "Not found", "description": "Not found"},
                "insurance_info": {"policy_number": "Not found", "coverage_type": "Not found", "deductible": "Not found"},
                "damage_details": {"damaged_parts": [], "estimated_cost": "Not found", "severity": "Not found"},
                "summary": "Failed to parse response",
                "extraction_confidence": "Low"
            }
            
    except Exception as e:
        st.error(f"Error extracting claim form details: {e}")
        return {
            "claimant_info": {"name": "Not found", "phone": "Not found", "email": "Not found", "address": "Not found"},
            "vehicle_info": {"make": "Not found", "model": "Not found", "year": "Not found", "vin": "Not found", "license_plate": "Not found"},
            "incident_info": {"date": "Not found", "time": "Not found", "location": "Not found", "description": "Not found"},
            "insurance_info": {"policy_number": "Not found", "coverage_type": "Not found", "deductible": "Not found"},
            "damage_details": {"damaged_parts": [], "estimated_cost": "Not found", "severity": "Not found"},
            "summary": f"API error: {str(e)}",
            "extraction_confidence": "Low"
        }

def generate_claim_form_pdf(extracted_data, form_filename, model_used):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    title_style = styles['Title']
    normal_style = styles['Normal']
    heading_style = styles['Heading2']
    
    elements.append(Paragraph("Claim Form Extraction Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"<b>Form File:</b> {form_filename}", normal_style))
    elements.append(Paragraph(f"<b>Model Used:</b> {model_used}", normal_style))
    elements.append(Paragraph(f"<b>Extraction Confidence:</b> {extracted_data.get('extraction_confidence', 'Unknown')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Claimant Information
    elements.append(Paragraph("<b>Claimant Information</b>", heading_style))
    claimant = extracted_data.get('claimant_info', {})
    elements.append(Paragraph(f"<b>Name:</b> {claimant.get('name', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Phone:</b> {claimant.get('phone', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Email:</b> {claimant.get('email', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Address:</b> {claimant.get('address', 'Not found')}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Vehicle Information
    elements.append(Paragraph("<b>Vehicle Information</b>", heading_style))
    vehicle = extracted_data.get('vehicle_info', {})
    elements.append(Paragraph(f"<b>Make:</b> {vehicle.get('make', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Model:</b> {vehicle.get('model', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Year:</b> {vehicle.get('year', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>VIN:</b> {vehicle.get('vin', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>License Plate:</b> {vehicle.get('license_plate', 'Not found')}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Incident Information
    elements.append(Paragraph("<b>Incident Information</b>", heading_style))
    incident = extracted_data.get('incident_info', {})
    elements.append(Paragraph(f"<b>Date:</b> {incident.get('date', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Time:</b> {incident.get('time', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Location:</b> {incident.get('location', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Description:</b> {incident.get('description', 'Not found')}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Insurance Information
    elements.append(Paragraph("<b>Insurance Information</b>", heading_style))
    insurance = extracted_data.get('insurance_info', {})
    elements.append(Paragraph(f"<b>Policy Number:</b> {insurance.get('policy_number', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Coverage Type:</b> {insurance.get('coverage_type', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Deductible:</b> {insurance.get('deductible', 'Not found')}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Damage Details
    elements.append(Paragraph("<b>Damage Details</b>", heading_style))
    damage = extracted_data.get('damage_details', {})
    elements.append(Paragraph(f"<b>Estimated Cost:</b> {damage.get('estimated_cost', 'Not found')}", normal_style))
    elements.append(Paragraph(f"<b>Severity:</b> {damage.get('severity', 'Not found')}", normal_style))
    
    damaged_parts = damage.get('damaged_parts', [])
    if damaged_parts:
        elements.append(Paragraph("<b>Damaged Parts:</b>", normal_style))
        for part in damaged_parts:
            elements.append(Paragraph(f"• {part}", normal_style))
    else:
        elements.append(Paragraph("<b>Damaged Parts:</b> None listed", normal_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    # Summary
    elements.append(Paragraph("<b>Summary</b>", heading_style))
    elements.append(Paragraph(extracted_data.get('summary', 'No summary available'), normal_style))
    
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def generate_image_damage_pdf(image_filename, detected_parts, summary, model_used):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    title_style = styles['Title']
    normal_style = styles['Normal']
    heading_style = styles['Heading2']
    elements.append(Paragraph("Image-based Car Damage Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"<b>Image File:</b> {image_filename}", normal_style))
    elements.append(Paragraph(f"<b>Model Used:</b> {model_used}", normal_style))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph(f"<b>Incident Summary:</b> {summary}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("<b>Detected Damaged Parts</b>", heading_style))
    if not detected_parts:
        elements.append(Paragraph("No damaged parts detected.", normal_style))
    else:
        for part, reason, severity in detected_parts:
            elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
            elements.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
            elements.append(Paragraph(f"<b>Severity:</b> {severity}", normal_style))
            elements.append(Spacer(1, 0.12*inch))
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Helper to get summary and severity using Azure OpenAI GPT-4o
def get_image_incident_summary_and_severity(parts, model_name="gpt-4o"):
    # Compose a prompt for summary and severity
    prompt = f"""
    Given the following detected damaged vehicle parts and reasons, write a 2-3 sentence summary describing how the incident might have happened. Then, for each part, estimate the severity (high/medium/low) based on typical accident scenarios. Return a JSON object with 'summary' and a 'parts' array of [part, reason, severity].
    PARTS: {parts}
    Format:
    {{
      "summary": "...",
      "parts": [["part", "reason", "severity"], ...]
    }}
    """
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert vehicle accident analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        text = response.choices[0].message.content
        if not text:
            return '', []
        import json
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                return result.get('summary', ''), result.get('parts', [])
            except Exception:
                return '', []
        return '', []
    except Exception as e:
        import streamlit as st
        st.error(f"Error generating summary/severity: {e}")
        return '', []

# --- Add the function from cause_of_loss.py ---
def extract_cause_of_loss_with_gpt_vision(image: Image.Image, model_name: str = "gpt-4o") -> str:
    """
    Analyze the uploaded form image using Azure OpenAI GPT-4 Vision model to extract cause of accident or accident description.
    Returns the cause of accident or accident description as a string.
    """
    # Convert image to bytes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file.name, format="PNG")
        tmp_file.seek(0)
        img_bytes = tmp_file.read()
    prompt = (
        "You are an expert insurance claim form analyzer. "
        "Analyze the uploaded insurance claim form image and extract the cause of accident or accident description. "
        "Look for fields labeled as: 'Cause of Loss', 'Accident Description', 'What Happened', 'Loss Description', 'Damage Description', 'How did the accident occur', 'Circumstances of Loss', etc. "
        "If you find multiple descriptions, prioritize the most detailed one. "
        "Return ONLY the extracted cause of accident or accident description as a clear, concise text. "
        "If no accident description is found, return 'No accident description found in the form'."
    )
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert insurance claim form analyzer."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")}}
                    ]
                }
            ],
            max_tokens=400
        )
        text = response.choices[0].message.content
        # Return the extracted text directly
        if not text:
            return "No accident description found in the form"
        return text.strip()
    except Exception as e:
        print(f"Error analyzing image with GPT-4 Vision: {e}")
        return "Error occurred while analyzing the form"

def generate_cause_of_loss_pdf(cause_of_loss: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, "Extracted Cause of Loss")
    c.setFont("Helvetica", 12)
    text_object = c.beginText(72, height - 110)
    for line in cause_of_loss.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Callback function for the download button
def generate_pdf_report_with_dr():
    potential_parts = st.session_state.get('potential_parts', [])
    damage_report_text = st.session_state['damage_report_text']
    policy_story = st.session_state['policy_story']
    estimate_text = st.session_state.get('estimate_text', "")
    file_name = st.session_state['damage_report_file_name']
    model_name = "gpt-4o"
    estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
    try:
        st.session_state['extracted_estimate_parts'] = estimate_parts
    except Exception:
        pass
    # Build reference reasons map from AI-interpreted potential parts
    try:
        ai_part_reasons = {}
        for p in (potential_parts or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                name = str(p[0]).strip()
                rsn = str(p[1]).strip()
                if name and rsn and name not in ai_part_reasons:
                    ai_part_reasons[name] = rsn
    except Exception:
        ai_part_reasons = {}
    result = filter_damage_report(damage_report_text, potential_parts, model_name, estimate_text, estimate_parts, ai_part_reasons=ai_part_reasons, claim_story_text=policy_story)
    
    matched_parts = result.get('relevant_damages', [])
    doubtful_parts = result.get('doubtful_damages', [])
    rejected_parts = result.get('rejected_parts', [])
    ai_interpreted_parts = {"claim_story": potential_parts} if potential_parts else None
    
    pdf_bytes = generate_matching_parts_pdf(
        policy_story="",
        matching_parts=matched_parts,
        summary=result.get('summary', ''),
        recommendation=result.get('recommendation', ''),
        damage_report_file=file_name,
        model_used=model_name,
        doubtful_parts=doubtful_parts,
        rejected_parts=rejected_parts,
        ai_interpreted_parts=ai_interpreted_parts,
    consistency_report=None,
    view_mode="with_dr"
    )
    st.session_state['pdf_data'] = pdf_bytes

def generate_pdf_report_without_dr():
    per_image_potential_parts = st.session_state.get('per_image_potential_parts', {})
    claim_description = st.session_state.get('claim_description', '') or st.session_state.get('policy_story', '')
    estimate_text = st.session_state.get('estimate_text', "")
    image_names = st.session_state.get('car_image_names', {})
    model_name = "gpt-4o"
    
    all_detected_parts = []
    for parts_list in per_image_potential_parts.values():
        all_detected_parts.extend([part[0] for part in parts_list])
    damage_report_text = "\n".join(all_detected_parts)

    potential_parts = extract_damage_parts_from_story(claim_description, model_name)
    estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
    # Build reference reasons map from story-extracted parts
    try:
        ai_part_reasons = {}
        for p in (potential_parts or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                name = str(p[0]).strip()
                rsn = str(p[1]).strip()
                if name and rsn and name not in ai_part_reasons:
                    ai_part_reasons[name] = rsn
    except Exception:
        ai_part_reasons = {}
    result = filter_damage_report(damage_report_text, potential_parts, model_name, estimate_text, estimate_parts, ai_part_reasons=ai_part_reasons, claim_story_text=claim_description)
    
    image_damage_dict = {k: [p[0] for p in v] for k, v in per_image_potential_parts.items()}
    # Build consistency prompt with estimate context and unobserved claimed parts
    estimate_text_ctx = estimate_text or ""
    estimate_parts_ctx = estimate_parts or []
    try:
        ai_detected_flat = []
        for _k, _v in per_image_potential_parts.items():
            for _t in _v:
                if isinstance(_t, (list, tuple)) and _t:
                    ai_detected_flat.append(str(_t[0]))
                elif isinstance(_t, str):
                    ai_detected_flat.append(_t)
        def _norm(s: str):
            return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
        ai_norms = {_norm(x) for x in ai_detected_flat if x}
        unobserved_claimed = []
        for ep in estimate_parts_ctx:
            if _norm(ep) and _norm(ep) not in ai_norms:
                unobserved_claimed.append(ep)
    except Exception:
        unobserved_claimed = []

    consistency_prompt = f"""
    You are an expert vehicle accident analyst. Analyze the damage distribution across multiple car images to validate consistency with a single accident event.

    Detected damaged parts from car images (AI-interpreted):
    {image_damage_dict}

    Accident description from claim form:
    {claim_description}

    Estimate copy (raw text):
    {estimate_text_ctx}

    Extracted parts from estimate copy (claimed/repaired):
    {estimate_parts_ctx}

    Parts claimed in estimate but NOT observed as damaged in images (unobserved claimed parts):
    {unobserved_claimed}

    Consistency checks:
    - Validate whether damages seen across multiple images align with a single accident.
    - If damage appears on left and right sides but not front/back, flag as potentially inconsistent (suggests multiple incidents).
    - Check if impact direction matches damage pattern (front collision should show front damage, not isolated rear damage).
    - Verify damage severity is consistent with described impact force and direction.
    - Look for contradictory damage patterns (e.g., both front and rear damage from a single-direction collision).
    - Assess if damage distribution follows logical impact physics and force transfer.
    - Check for inconsistent damage patterns within the same area: if adjacent parts show damage but connecting parts are undamaged, flag as potentially inconsistent (e.g., both left and right front headlights damaged but front bumper and hood are intact, suggesting separate incidents rather than a single collision).
    - Compare estimate parts vs. AI-observed damages: explain whether any unobserved claimed parts are plausibly hidden/internal from the described impact or suggest unrelated/previous damage.

    Provide a concise 3–4 line assessment in measured, diplomatic language. Avoid using single-word labels like "consistent" or "inconsistent". End with a neutral line beginning with "Overall assessment:" that summarizes alignment and any exceptions.
    """
    try:
        consistency_response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert vehicle accident analyst."},
                {"role": "user", "content": consistency_prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        consistency_report = consistency_response.choices[0].message.content.strip()
    except Exception as e:
        consistency_report = f"Error checking consistency: {e}"

    combined_image_names = ", ".join([name for name in image_names.values()])
    
    pdf_bytes = generate_matching_parts_pdf(
        policy_story="",
        matching_parts=result.get('relevant_damages', []),
        summary=result.get('summary', ''),
        recommendation=result.get('recommendation', ''),
        damage_report_file=combined_image_names,
        model_used=model_name,
        doubtful_parts=result.get('doubtful_damages', []),
        rejected_parts=result.get('rejected_parts', []),
        ai_interpreted_parts=per_image_potential_parts,
    consistency_report=consistency_report,
    view_mode="without_dr"
    )
    st.session_state['pdf_data'] = pdf_bytes


def main():
    # Always show the Without Damage Report workflow for now
    without_damage_report_workflow()

def with_damage_report_workflow():

    st.markdown('<h2 class="main-header">Car Damage Analysis (With Damage Report)</h2>', unsafe_allow_html=True)
    st.write("")
    # Removed step-by-step instructions from the final page as requested
    st.markdown("---")

    # Step 1: Upload Damage Report
    if 'damage_report_uploaded' not in st.session_state:
        st.session_state['damage_report_uploaded'] = False
    if 'damage_report_text' not in st.session_state:
        st.session_state['damage_report_text'] = ""
    if 'damage_report_file_name' not in st.session_state:
        st.session_state['damage_report_file_name'] = ""

    if not st.session_state['damage_report_uploaded']:
        st.markdown('<h5 class="section-header">📄 Step 1: Upload Damage Report</h5>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a text or PDF file",
            type=['txt', 'pdf'],
            help="Upload your damage report as a text or PDF file",
            key="damage_report_upload"
        )
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                damage_report_text = extract_text_from_pdf(uploaded_file)
                st.session_state['damage_report_text'] = damage_report_text
                st.session_state['damage_report_file_name'] = uploaded_file.name
                st.success(f"✅ PDF file uploaded: {uploaded_file.name}")
                with st.expander("📋 Damage Report PDF Preview"):
                    st.text_area("Extracted Report Content", damage_report_text, height=200, disabled=True)
            else:
                damage_report_text = uploaded_file.read().decode('utf-8')
                st.session_state['damage_report_text'] = damage_report_text
                st.session_state['damage_report_file_name'] = uploaded_file.name
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                with st.expander("📋 Damage Report Preview"):
                    st.text_area("Report Content", damage_report_text, height=200, disabled=True)
            if st.button("Upload Damage Report", type="primary", use_container_width=True):
                st.session_state['damage_report_uploaded'] = True
                st.rerun()
        return

    # Step 2: Upload Claim Form
    if 'claim_form_uploaded' not in st.session_state:
        st.session_state['claim_form_uploaded'] = False
    if 'policy_story' not in st.session_state:
        st.session_state['policy_story'] = ""
    if 'cause_of_loss' not in st.session_state:
        st.session_state['cause_of_loss'] = ""

    if not st.session_state['claim_form_uploaded']:
        st.markdown('<h5 class="section-header">📝 Step 2: Upload Claim Form</h5>', unsafe_allow_html=True)
        story_file = st.file_uploader("Choose a PDF or image file", type=['pdf', 'jpg', 'jpeg', 'png'], key="claim_form_upload")
        if story_file:
            if story_file.type == "application/pdf" or story_file.name.lower().endswith(".pdf"):
                pdf_text = extract_text_from_pdf(story_file)
                st.session_state['policy_story'] = pdf_text
                st.success(f"✅ PDF Uploaded: {story_file.name}")
                with st.expander("📋 PDF Content Preview"):
                    st.text_area("Extracted Text", pdf_text, height=200, disabled=True)
                with st.spinner("Extracting cause of loss from PDF..."):
                    cause_of_loss = extract_cause_of_loss(pdf_text)
                st.session_state['cause_of_loss'] = cause_of_loss
                st.info(f"**Extracted Cause of Loss:** {cause_of_loss}")
            else:
                claim_image = Image.open(story_file)
                st.success(f"✅ Image Uploaded: {story_file.name}")
                st.image(claim_image, caption="Uploaded Claim Form Image", use_container_width=True)
                with st.spinner("Extracting cause of loss from image..."):
                    cause_of_loss = extract_cause_of_loss_with_gpt_vision(claim_image)
                st.session_state['cause_of_loss'] = cause_of_loss
                st.info(f"**Extracted Cause of Loss:** {cause_of_loss}")
                st.session_state['policy_story'] = cause_of_loss
            if st.button("Upload Claim Form", type="primary", use_container_width=True):
                st.session_state['claim_form_uploaded'] = True
                st.rerun()
        return

    # Step 3: Upload Estimate Copy
    if 'estimate_uploaded' not in st.session_state:
        st.session_state['estimate_uploaded'] = False
    if 'estimate_text' not in st.session_state:
        st.session_state['estimate_text'] = ""
    if 'estimate_file_name' not in st.session_state:
        st.session_state['estimate_file_name'] = ""

    if not st.session_state['estimate_uploaded']:
        st.markdown('<h5 class="section-header">💰 Step 3: Upload Estimate Copy (Required)</h5>', unsafe_allow_html=True)
        uploaded_estimate = st.file_uploader(
            "Choose an estimate copy as PDF or text file",
            type=['txt', 'pdf'],
            help="Upload your repair estimate as a text or PDF file (required)",
            key="estimate_upload_with_report"
        )
        
        if uploaded_estimate is not None:
            if uploaded_estimate.type == "application/pdf" or uploaded_estimate.name.lower().endswith(".pdf"):
                estimate_text = extract_text_from_pdf(uploaded_estimate)
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ PDF estimate uploaded: {uploaded_estimate.name}")
            else:
                estimate_text = uploaded_estimate.read().decode('utf-8')
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ Text estimate uploaded: {uploaded_estimate.name}")
            
            with st.spinner("🔍 Extracting parts from estimate copy..."):
                model_name = "gpt-4o"
                estimate_parts = extract_parts_from_estimate_copy(st.session_state['estimate_text'], model_name)
                st.session_state['extracted_estimate_parts'] = estimate_parts

            if st.session_state.get('extracted_estimate_parts'):
                st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)
                st.markdown('<h4 class="section-header">📋 Extracted Estimate Parts</h4>', unsafe_allow_html=True)
                st.success("✅ Successfully extracted vehicle parts from estimate copy!")
                
                # Display extracted parts in a clean format
                cols = st.columns(3)
                for i, ep in enumerate(st.session_state['extracted_estimate_parts']):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #007bff;'>
                            <strong>🔧 {str(ep)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.info(f"📊 **Total parts found:** {len(st.session_state['extracted_estimate_parts'])}")
                st.markdown('<div style="height: 15px"></div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No parts were extracted from the estimate copy. Please check the file content.")
        
        # Require estimate and extracted parts before proceeding
        if uploaded_estimate is not None:
            extracted_parts = st.session_state.get('extracted_estimate_parts', [])
            if extracted_parts and len(extracted_parts) > 0:
                if st.button("Analyze Reports and Forms", type="primary", use_container_width=True):
                    st.session_state['estimate_uploaded'] = True
                    st.rerun()
            else:
                st.error("⚠️ Cannot proceed: No vehicle parts were extracted from the estimate copy. Please upload a valid estimate document with vehicle parts listed.")
        else:
            st.warning("⚠️ Please upload an estimate copy to proceed with the analysis.")
        return

    # Step 4: Analysis and Results
    damage_report_text = st.session_state['damage_report_text']
    policy_story = st.session_state['policy_story']
    estimate_text = st.session_state.get('estimate_text', "")
    file_name = st.session_state['damage_report_file_name']
    model_name = "gpt-4o"

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        st.error("Azure OpenAI API key not configured. Please set AZURE_OPENAI_API_KEY in your environment.")
        return

    with st.spinner("🤖 Extracting potential damage parts from claim form..."):
        potential_parts = extract_damage_parts_from_story(policy_story, model_name)
        st.session_state['potential_parts'] = potential_parts
        # Show extracted claim form text for debugging
        if not potential_parts:
            st.error("Could not extract damage parts from the claim form. Please try again.")
            st.markdown("**Extracted Claim Form Text:**")
            st.code(policy_story if policy_story else "[Empty claim form text]", language="text")
            st.markdown("**LLM Response (Debug):**")
            # Try to show the intermediate reasoning and summary if available
            try:
                debug_result = extract_damage_parts_with_reasoning(policy_story, model_name)
                st.json(debug_result)
            except Exception as e:
                st.warning(f"Error getting debug info: {e}")
            return

    with st.spinner("🔍 Extracting parts from estimate copy..."):
        estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
        # Cache for display on the main page
        try:
            st.session_state['extracted_estimate_parts'] = estimate_parts
        except Exception:
            pass

    with st.spinner("🔍 Filtering damage report based on claim form..."):
        # Use AI-interpreted parts from claim story as the authoritative set
        try:
            ai_detected_parts = [ (p[0] if isinstance(p, (list, tuple)) else p) for p in (potential_parts or []) ]
        except Exception:
            ai_detected_parts = []
        # Build reference reasons map again for consistent prompting
        try:
            ai_part_reasons = {}
            for p in (potential_parts or []):
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    name = str(p[0]).strip()
                    rsn = str(p[1]).strip()
                    if name and rsn and name not in ai_part_reasons:
                        ai_part_reasons[name] = rsn
        except Exception:
            ai_part_reasons = {}
        result = filter_damage_report(
            damage_report_text,
            potential_parts,
            model_name,
            estimate_text,
            estimate_parts,
            ai_detected_parts=ai_detected_parts,
            ai_part_reasons=ai_part_reasons,
            claim_story_text=policy_story
        )

    # Precompute PDF once to avoid rerun on download
    try:
        estimate_text = st.session_state.get('estimate_text', "")
        estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
        # Build a simple consistency check using the textual damage report
        try:
            # Compute parts from damage report that overlap with potential parts for a coarse signal
            dr_text = damage_report_text or ""
            potential_names = [ (p[0] if isinstance(p, (list, tuple)) else str(p)) for p in (potential_parts or []) ]
            missing_in_report = []
            for name in potential_names:
                if name and name.lower() not in (dr_text.lower()):
                    missing_in_report.append(name)
            consistency_prompt = f"""
            You are an expert vehicle accident analyst. Validate if this damage report aligns with the claim story.

            Damage report (text):
            {damage_report_text}

            Claim story:
            {policy_story}

            Extracted parts from estimate copy (claimed/repaired):
            {estimate_parts}

            Claim-story likely parts not mentioned in the damage report (may be hidden/unrelated):
            {missing_in_report}

            Provide a concise 3–4 line assessment.
            """
            consistency_response = azure_openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "You are an expert vehicle accident analyst."},
                    {"role": "user", "content": consistency_prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            consistency_report = consistency_response.choices[0].message.content.strip()
        except Exception as e:
            consistency_report = f"Error checking consistency: {e}"
        pdf_bytes = generate_matching_parts_pdf(
            policy_story="",
            matching_parts=result.get('relevant_damages', []),
            summary=result.get('summary', ''),
            recommendation=result.get('recommendation', ''),
            damage_report_file=file_name,
            model_used=model_name,
            doubtful_parts=result.get('doubtful_damages', []),
            rejected_parts=result.get('rejected_parts', []),
            ai_interpreted_parts={"claim_story": potential_parts} if potential_parts else None,
            consistency_report=consistency_report,
            view_mode="with_dr"
        )
        st.session_state['pdf_data'] = pdf_bytes
        # Store for display section parity with Without Damage Report
        try:
            st.session_state['consistency_report'] = consistency_report
        except Exception:
            pass
    except Exception:
        pass
    display_analysis_results(potential_parts, result, file_name, model_name, workflow="with_dr")

def without_damage_report_workflow():

    st.markdown('<h2 class="main-header">Car Damage Analysis (Without Damage Report)</h2>', unsafe_allow_html=True)
    st.write("")
    # Step 1: Upload up to 4 car images
    if 'car_images_uploaded' not in st.session_state:
        st.session_state['car_images_uploaded'] = False
    if 'car_images' not in st.session_state:
        st.session_state['car_images'] = {}
    if 'car_image_names' not in st.session_state:
        st.session_state['car_image_names'] = {}
    if not st.session_state['car_images_uploaded']:
        st.markdown('<h5 class="section-header">🖼️ Step 1: Upload Car Images (Front, Back, Left, Right)</h5>', unsafe_allow_html=True)
        col_front, col_back, col_left, col_right = st.columns(4)
        image_keys = ['front', 'back', 'left', 'right']
        uploaded = False
        for idx, (col, key) in enumerate(zip([col_front, col_back, col_left, col_right], image_keys)):
            with col:
                uploaded_file = st.file_uploader(f"{key.title()} Image", type=["jpg", "jpeg", "png"], key=f"car_image_{key}")
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.session_state['car_images'][key] = image
                    st.session_state['car_image_names'][key] = uploaded_file.name
                    st.image(image, caption=f"{key.title()} Image", use_container_width=True)
                    uploaded = True
        if st.button("Upload Selected Images", type="primary", use_container_width=True):
            if len(st.session_state['car_images']) > 0:
                st.session_state['car_images_uploaded'] = True
                st.rerun()
            else:
                st.warning("Please upload at least one image.")
        return

    # Step 2: Upload Claim Form
    if 'claim_form_uploaded' not in st.session_state:
        st.session_state['claim_form_uploaded'] = False
    if 'claim_description' not in st.session_state:
        st.session_state['claim_description'] = None
    if not st.session_state['claim_form_uploaded']:
        st.markdown('<h5 class="section-header">📝 Step 2: Upload Claim Form</h5>', unsafe_allow_html=True)
        uploaded_claim_form = st.file_uploader(
            "Choose a claim form image or PDF (jpg, png, pdf)",
            type=["jpg", "jpeg", "png", "pdf"],
            key="combined_claim_form_upload"
        )
        if uploaded_claim_form is not None:
            st.success(f"✅ Claim form uploaded: {uploaded_claim_form.name}")
            if uploaded_claim_form.type == "application/pdf" or uploaded_claim_form.name.lower().endswith(".pdf"):
                pdf_text = extract_text_from_pdf(uploaded_claim_form)
                with st.expander("📋 Claim Form PDF Preview"):
                    st.text_area("Extracted Text", pdf_text, height=200, disabled=True)
                with st.spinner("Extracting accident description from claim form PDF..."):
                    claim_description = extract_cause_of_loss(pdf_text)
                st.session_state['claim_description'] = claim_description
                st.info(f"**Extracted Description:** {claim_description}")
            else:
                claim_image = Image.open(uploaded_claim_form)
                with st.spinner("Extracting accident description from claim form image..."):
                    claim_description = extract_cause_of_loss_with_gpt_vision(claim_image)
                st.session_state['claim_description'] = claim_description
                st.info(f"**Extracted Description:** {claim_description}")
            # Show single button for proceeding to next step after claim form upload
            if st.session_state['claim_description']:
                if st.button("Upload Claim Form", type="primary", use_container_width=True):
                    st.session_state['claim_form_uploaded'] = True
                    st.rerun()
        return

    # Step 3: Upload Estimate Copy
    if 'estimate_uploaded' not in st.session_state:
        st.session_state['estimate_uploaded'] = False
    if 'estimate_text' not in st.session_state:
        st.session_state['estimate_text'] = ""
    if 'estimate_file_name' not in st.session_state:
        st.session_state['estimate_file_name'] = ""

    if not st.session_state['estimate_uploaded']:
        st.markdown('<h5 class="section-header">💰 Step 3: Upload Estimate Copy (Required)</h5>', unsafe_allow_html=True)
        uploaded_estimate = st.file_uploader(
            "Choose an estimate copy as PDF or text file",
            type=['txt', 'pdf'],
            help="Upload your repair estimate as a text or PDF file (required)",
            key="estimate_upload"
        )
        
        if uploaded_estimate is not None:
            if uploaded_estimate.type == "application/pdf" or uploaded_estimate.name.lower().endswith(".pdf"):
                estimate_text = extract_text_from_pdf(uploaded_estimate)
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ PDF estimate uploaded: {uploaded_estimate.name}")
            else:
                estimate_text = uploaded_estimate.read().decode('utf-8')
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ Text estimate uploaded: {uploaded_estimate.name}")
        
            with st.spinner("🔍 Extracting parts from estimate copy..."):
                model_name = "gpt-4o"
                estimate_parts = extract_parts_from_estimate_copy(st.session_state['estimate_text'], model_name)
                st.session_state['extracted_estimate_parts'] = estimate_parts

            if st.session_state.get('extracted_estimate_parts'):
                st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)
                st.markdown('<h4 class="section-header">📋 Extracted Estimate Parts</h4>', unsafe_allow_html=True)
                st.success("✅ Successfully extracted vehicle parts from estimate copy!")
                
                # Display extracted parts in a clean format
                cols = st.columns(3)
                for i, ep in enumerate(st.session_state['extracted_estimate_parts']):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #007bff;'>
                            <strong>🔧 {str(ep)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.info(f"📊 **Total parts found:** {len(st.session_state['extracted_estimate_parts'])}")
                st.markdown('<div style="height: 15px"></div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No parts were extracted from the estimate copy. Please check the file content.")

        # Require estimate and extracted parts before proceeding
        if uploaded_estimate is not None:
            extracted_parts = st.session_state.get('extracted_estimate_parts', [])
            if extracted_parts and len(extracted_parts) > 0:
                if st.button("Analyze Images and Forms", type="primary", use_container_width=True):
                    st.session_state['estimate_uploaded'] = True
                    st.rerun()
            else:
                st.error("⚠️ Cannot proceed: No vehicle parts were extracted from the estimate copy. Please upload a valid estimate document with vehicle parts listed.")
        else:
            st.warning("⚠️ Please upload an estimate copy to proceed with the analysis.")
        return

    # Step 4: Analysis and Results
    car_images = st.session_state['car_images']
    claim_description = st.session_state['claim_description']
    estimate_text = st.session_state.get('estimate_text', "")
    image_names = st.session_state.get('car_image_names', {})
    if st.session_state['estimate_uploaded'] and len(car_images) > 0 and claim_description:
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            st.error("Azure OpenAI API key not configured. Please set AZURE_OPENAI_API_KEY in your environment.")
            return
        # Analyze each image separately for AI-Interpreted Potentially Impacted Parts
        per_image_potential_parts = {}
        all_detected_parts = []
        model_name = "gpt-4o"
        with st.spinner("Extracting damage report text from car images..."):
            for key, image in car_images.items():
                detected_parts = analyze_image_with_gpt_vision(image, "gpt-4o")
                per_image_potential_parts[key] = detected_parts
                for part_tuple in detected_parts:
                    all_detected_parts.append(part_tuple[0])
        st.session_state['per_image_potential_parts'] = per_image_potential_parts
        # Combine all detected parts into a single damage report text
        damage_report_text = "\n".join(all_detected_parts)
        with st.spinner("Inferring potential damages from description..."):
            potential_parts = extract_damage_parts_from_story(claim_description, model_name)
        with st.spinner("🔍 Extracting parts from estimate copy..."):
            estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
            # Cache for reuse in consistency prompts without extra calls
            try:
                st.session_state['extracted_estimate_parts'] = estimate_parts
            except Exception:
                pass
        with st.spinner("Filtering damage report based on description..."):
            ai_detected_parts = []
            try:
                for lst in per_image_potential_parts.values():
                    for tup in lst:
                        if isinstance(tup, (list, tuple)) and tup:
                            ai_detected_parts.append(str(tup[0]))
                        elif isinstance(tup, str):
                            ai_detected_parts.append(tup)
            except Exception:
                pass
            # Build reasons map from per-image interpreted parts first; fallback to story potential parts
            try:
                ai_part_reasons = {}
                for lst in per_image_potential_parts.values():
                    for tup in lst:
                        if isinstance(tup, (list, tuple)) and len(tup) >= 2:
                            name = str(tup[0]).strip()
                            rsn = str(tup[1]).strip()
                            if name and rsn and name not in ai_part_reasons:
                                ai_part_reasons[name] = rsn
                if not ai_part_reasons:
                    for p in (potential_parts or []):
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            name = str(p[0]).strip()
                            rsn = str(p[1]).strip()
                            if name and rsn and name not in ai_part_reasons:
                                ai_part_reasons[name] = rsn
            except Exception:
                ai_part_reasons = {}
            result = filter_damage_report(damage_report_text, potential_parts, model_name, estimate_text, estimate_parts, ai_detected_parts=ai_detected_parts, ai_part_reasons=ai_part_reasons, claim_story_text=claim_description)
            # Remove additional damages from result
            if 'additional_damages' in result:
                result['additional_damages'] = []
            # Ensure rejected_parts field exists
            if 'rejected_parts' not in result:
                result['rejected_parts'] = []
            # Remove missing_parts if it exists (now combined with rejected_parts)
            if 'missing_parts' in result:
                del result['missing_parts']

        # Restrict matched damages to only those present in AI-interpreted (image) parts
        def _normalize_part_name(name: str):
            return re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip()

        try:
            ai_parts = []
            for lst in per_image_potential_parts.values():
                for tup in lst:
                    if isinstance(tup, (list, tuple)) and tup:
                        ai_parts.append(str(tup[0]))
                    elif isinstance(tup, str):
                        ai_parts.append(tup)
            ai_norms = [_normalize_part_name(p) for p in ai_parts if p]
            def _matches_ai(part: str):
                pn = _normalize_part_name(part)
                if not pn:
                    return False
                for an in ai_norms:
                    if an == pn or an in pn or pn in an:
                        return True
                return False
            result['relevant_damages'] = [d for d in result.get('relevant_damages', []) if _matches_ai(d.get('part', ''))]
        except Exception:
            pass
        combined_image_names = ", ".join([name for name in image_names.values()])
        # Compute the same Accident Consistency Check shown on the page
        image_damage_dict = {k: [p[0] for p in v] for k, v in per_image_potential_parts.items()}
        claim_form_story = st.session_state.get('claim_description', '') or st.session_state.get('policy_story', '')
        estimate_text_ctx = st.session_state.get('estimate_text', "")
        estimate_parts_ctx = st.session_state.get('extracted_estimate_parts', []) or estimate_parts
        try:
            ai_detected_flat = []
            for _k, _v in per_image_potential_parts.items():
                for _t in _v:
                    if isinstance(_t, (list, tuple)) and _t:
                        ai_detected_flat.append(str(_t[0]))
                    elif isinstance(_t, str):
                        ai_detected_flat.append(_t)
            def _norm(s: str):
                return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
            ai_norms = {_norm(x) for x in ai_detected_flat if x}
            unobserved_claimed = []
            for ep in estimate_parts_ctx or []:
                if _norm(ep) and _norm(ep) not in ai_norms:
                    unobserved_claimed.append(ep)
        except Exception:
            unobserved_claimed = []

        consistency_prompt = f"""
        You are an expert vehicle accident analyst. Analyze consistency using a strict 2-step process and output only a concise 3–4 line assessment.

        Detected damaged parts from car images (AI-interpreted):
        {image_damage_dict}

        Accident description from claim form:
        {claim_form_story}

        Estimate copy (raw text):
        {estimate_text_ctx}

        Extracted parts from estimate copy (claimed/repaired):
        {estimate_parts_ctx}

        Parts claimed in estimate but NOT observed as damaged in images (unobserved claimed parts):
        {unobserved_claimed}

        Follow this strict 3-step process:
        1) Compare Estimate Copy parts with the Claim Story:
           - If the claim story describes a collision type (e.g., front collision) and the estimate lists relevant parts (e.g., front bumper, fender), consider that consistent.
           - If the estimate parts do not align with the described collision type or location, mark inconsistent with a brief reason tied to physics/direction.
        2) Compare Estimate Copy parts with the Damage Report (images/text):
           - If the damage report shows extra damaged areas not included in the estimate, do NOT mark inconsistent based only on that; missing items in the estimate are acceptable.
           - Mark inconsistent only if the estimate claims repairs that contradict the damage report or the claim story (e.g., estimate claims rear repairs but images/story show only a clear front impact with no rear damage).
           - If an estimate part is not visible in images but could reasonably be hidden/internal based on the impact, treat as plausible; do not mark inconsistent solely for non-visibility—note uncertainty briefly.
        3) Check for inconsistent damage patterns within the same area:
           - If adjacent parts show damage but connecting parts are undamaged, flag as potentially inconsistent (e.g., both left and right front headlights damaged but front bumper and hood are intact, suggesting separate incidents rather than a single collision).
           - Verify that the damage pattern follows logical impact physics and force transfer for a single accident.

        Provide a concise 3–4 line assessment in measured language. Summarize alignment and any exceptions from Steps 1 and 2".
        """
        try:
            consistency_response = azure_openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "You are an expert vehicle accident analyst."},
                    {"role": "user", "content": consistency_prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            consistency_report = consistency_response.choices[0].message.content.strip()
        except Exception as e:
            consistency_report = f"Error checking consistency: {e}"

        # Precompute PDF for download to avoid rerun (now includes the consistency section)
        try:
            pdf_bytes = generate_matching_parts_pdf(
                policy_story="",
                matching_parts=result.get('relevant_damages', []),
                summary=result.get('summary', ''),
                recommendation=result.get('recommendation', ''),
                damage_report_file=combined_image_names,
                model_used=model_name,
                doubtful_parts=result.get('doubtful_damages', []),
                rejected_parts=result.get('rejected_parts', []),
                ai_interpreted_parts=per_image_potential_parts,
                consistency_report=consistency_report
            )
            st.session_state['pdf_data'] = pdf_bytes
        except Exception:
            pass
        display_combined_analysis(per_image_potential_parts, potential_parts, result, combined_image_names, model_name, consistency_report=consistency_report, claim_description=claim_description, ai_detected_parts=ai_detected_parts)

def display_combined_analysis(per_image_potential_parts, potential_parts, result, file_name, model_name, consistency_report: str | None = None, claim_description: str = "", ai_detected_parts: list = None):
    """Display 4 sections for AI-Interpreted Potentially Impacted Parts, rest combined"""
    st.markdown('<div style="height: 24px"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header" style="margin-bottom: 2.5rem;">🔎 Analysis Results</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)
    # AI-Interpreted Potentially Impacted Parts by image
    st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">🤖 AI-Interpreted Potentially Impacted Parts</h3>', unsafe_allow_html=True)
    for key in ['front', 'back', 'left', 'right']:
        if key in per_image_potential_parts:
            detected_parts = per_image_potential_parts[key]
            st.markdown(f'<h4 class="section-header" style="margin-bottom: 1.2rem;">{key.title()} Image</h4>', unsafe_allow_html=True)
            st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)
            parts_cols = st.columns(3)
            for i, part_tuple in enumerate(detected_parts):
                part = part_tuple[0]
                reason = part_tuple[1] if len(part_tuple) > 1 else ""
                with parts_cols[i % 3]:
                    st.markdown(f'''
                    <div class="damage-item" style="margin-bottom: 18px;">
                        <div>🔧 {part}</div>
                        {f"<div class='reason-on-hover'>Reason: {reason}</div>" if reason else ''}
                    </div>
                    ''', unsafe_allow_html=True)
            st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

    # Summary & Recommendation (kept in original order here)
    st.markdown('<div style="height: 6px"></div>', unsafe_allow_html=True)
    sum_col1, sum_col2 = st.columns([2,1])
    with sum_col1:
        st.markdown(f"""
        <div class='result-box'>
            <div style='font-size:1.05rem;'><b>Summary</b></div>
            <div style='color:var(--muted); margin-top:6px;'>{result.get('summary','')}</div>
        </div>
        """, unsafe_allow_html=True)
    with sum_col2:
        recommendation_color = {
            "Approve": "🟢",
            "Investigate": "🟡",
            "Deny": "🔴"
        }.get(result.get('recommendation'), "⚪")
        st.markdown(f"""
        <div class='result-box'>
            <div style='font-size:1.05rem;'><b>Recommendation</b></div>
            <div style='margin-top:6px;'>{recommendation_color} {result.get('recommendation','')}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div style="height: 14px"></div>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none; border-top:1px solid var(--border);"/>', unsafe_allow_html=True)
    st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)
    # Extracted Estimate Parts
    estimate_parts_display = st.session_state.get('extracted_estimate_parts', [])
    if estimate_parts_display:
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.2rem;">🧾 Extracted Estimate Parts</h3>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, ep in enumerate(estimate_parts_display):
            with cols[i % 3]:
                st.markdown(f"""
                <div class='result-box' style='margin-bottom: 10px;'>
                    • {str(ep)}
                </div>
                """, unsafe_allow_html=True)
        st.markdown('<div style="height: 14px"></div>', unsafe_allow_html=True)
    # Relevant damages (combined)
    if result.get('relevant_damages'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">✅ Matched Damages</h3>', unsafe_allow_html=True)
        for damage in result['relevant_damages']:
            part = damage.get('part', '').title()
            desc = damage.get('damage_description', '')
            reason = damage.get('reason', '')
            st.markdown(f'''
            <div class="damage-item" style="margin-bottom: 12px;">
                <div><strong>{part}</strong></div>
                <div style='margin-top:4px;'>{desc}</div>
                {f"<div class='reason-on-hover'>Reason: {reason}</div>" if reason else ''}
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # Doubtful damages (combined)
    if result.get('doubtful_damages'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">❔ Doubtful Damages</h3>', unsafe_allow_html=True)
        for damage in result['doubtful_damages']:
            reason = damage.get('reason', '')
            st.markdown(f'''
            <div class="damage-item" style="margin-bottom: 12px;">
                <div><strong>{damage.get('part', '').title()}</strong></div>
                <div style='margin-top:4px;'>{damage.get('damage_description', '')}</div>
                {f"<div class='reason-on-hover'>Reason: {reason}</div>" if reason else ''}
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # Rejected parts (combined)
    if result.get('rejected_parts'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">🚫 Rejected Parts</h3>', unsafe_allow_html=True)
        for rejected in result['rejected_parts']:
            st.markdown(f'''
            <div class="rejected-item" style="margin-bottom: 14px;">
                <div><strong>🚫 {rejected.get("part", "")}</strong></div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # --- Consistency Check LLM Call ---
    st.markdown('<div style="height: 32px"></div>', unsafe_allow_html=True)
    if consistency_report is None:
        # Compute only if not provided by caller
        image_damage_dict = {k: [p[0] for p in v] for k, v in per_image_potential_parts.items()}
        claim_form_story = st.session_state.get('claim_description', '') or st.session_state.get('policy_story', '')
        # Bring in estimate context (text + extracted parts) and compute unobserved claimed parts
        estimate_text_ctx = st.session_state.get('estimate_text', "")
        estimate_parts_ctx = st.session_state.get('extracted_estimate_parts', [])
        try:
            ai_detected_flat = []
            for _k, _v in per_image_potential_parts.items():
                for _t in _v:
                    if isinstance(_t, (list, tuple)) and _t:
                        ai_detected_flat.append(str(_t[0]))
                    elif isinstance(_t, str):
                        ai_detected_flat.append(_t)
            def _norm(s: str):
                return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
            ai_norms = {_norm(x) for x in ai_detected_flat if x}
            unobserved_claimed = []
            for ep in estimate_parts_ctx or []:
                if _norm(ep) and _norm(ep) not in ai_norms:
                    unobserved_claimed.append(ep)
        except Exception:
            unobserved_claimed = []

        consistency_prompt = f"""
        You are an expert vehicle accident analyst. Analyze consistency using a strict 3-step process and output a concise 4-5 line assessment.

        Detected damaged parts from car images (AI-interpreted):
        {image_damage_dict}

        Accident description from claim form:
        {claim_form_story}

        Estimate copy (raw text):
        {estimate_text_ctx}

        Extracted parts from estimate copy (claimed/repaired):
        {estimate_parts_ctx}

        Parts claimed in estimate but NOT observed as damaged in images (unobserved claimed parts):
        {unobserved_claimed}

        Follow this strict 3-step process:
        1) Compare Estimate Copy parts with the Claim Story:
           - If the claim story describes a collision type (e.g., front collision) and the estimate lists relevant parts (e.g., front bumper, fender), consider that consistent.
           - If the estimate parts do not align with the described collision type or location, mark inconsistent with a brief reason tied to physics/direction.
        2) Compare Estimate Copy parts with the Damage Report (images/text):
           - If the damage report shows extra damaged areas not included in the estimate, do not mark inconsistent based only on that; mention that this damage is likely due to a prior incident.
           - Mark inconsistent only if the estimate claims repairs that contradict the damage report or the claim story (e.g., estimate claims rear repairs but images/story show only a clear front impact with no rear damage).
           - If an estimate part is not visible in images but could plausibly be hidden/internal based on the impact, treat as plausible.
        3) Check for inconsistent damage patterns within the same area:
           - If adjacent parts show damage but connecting parts are undamaged, flag as potentially inconsistent (e.g., both left and right front headlights damaged but the front bumper and hood are intact, suggesting separate incidents).
           - SINGLE ACCIDENT CONSISTENCY: For a single front collision, damage should follow a logical impact pattern from the center outward. If both left and right headlights are damaged but the central grille/bumper is undamaged, this indicates inconsistency.
           - Verify that the damage pattern follows a logical force transfer for a single accident as per the claim story and explicitly state if the damage pattern is inconsistent with a single collision scenario.

        Provide a concise 4-5 line assessment in measured language. Include a specific line explaining your reasoning for step 3 in the end.
        """
        try:
            consistency_response = azure_openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "You are an expert vehicle accident analyst."},
                    {"role": "user", "content": consistency_prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            consistency_report = consistency_response.choices[0].message.content.strip()
        except Exception as e:
            consistency_report = f"Error checking consistency: {e}"



    st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">🧩 Accident Consistency Check</h3>', unsafe_allow_html=True)
    st.info(consistency_report)
    
    # Add visual consistency check using GPT Vision
    if claim_description and st.session_state.get('car_images'):
        try:
            with st.spinner("Analyzing damage patterns with AI Vision..."):
                visual_consistency = analyze_single_accident_consistency_with_vision(
                    car_images=st.session_state.get('car_images', {}),
                    claim_story=claim_description,
                    ai_detected_parts=ai_detected_parts or [],
                    model_name=model_name
                )
            
            # Display visual consistency analysis
            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            st.markdown('<h4 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #1f77b4;">🔍 Visual Damage Pattern Analysis</h4>', unsafe_allow_html=True)
            
            # Color code based on result
            if visual_consistency.startswith("CONSISTENT"):
                st.success(f"✅ {visual_consistency}")
            elif visual_consistency.startswith("INCONSISTENT"):
                st.error(f"⚠️ {visual_consistency}")
            else:
                st.info(f"🔎 {visual_consistency}")
                
        except Exception as e:
            st.warning(f"⚠️ Could not perform visual damage analysis: {str(e)}")
    
    st.markdown('<div style="height: 32px"></div>', unsafe_allow_html=True)
    # Download results (combined)
    st.markdown("---")
    st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    st.markdown("### 📥 Download Final Report (PDF)")
    
    if 'pdf_data' not in st.session_state:
        st.session_state['pdf_data'] = None

    render_pdf_download_link(st.session_state.get('pdf_data', b''), 'damage_report_matched_parts.pdf')

def display_analysis_results(potential_parts, result, file_name, model_name, workflow="with_dr"):
    """Display the analysis results for both workflows"""
    st.markdown('<h2 class="section-header">🔎 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Potential parts section
    st.markdown('<h3 class="section-header">🤖 AI-Interpreted Potentially Impacted Parts</h3>', unsafe_allow_html=True)
    parts_cols = st.columns(3)
    for i, part_tuple in enumerate(potential_parts):
        part = part_tuple[0]
        reason = part_tuple[1] if len(part_tuple) > 1 else ""
        with parts_cols[i % 3]:
            try:
                safe_reason = re.sub(r"<[^>]+>", "", reason or "")
            except Exception:
                safe_reason = reason
            st.markdown(f'''
            <div class="damage-item">
                <div>🔧 {part}</div>
                {f"<div class='reason-on-hover'>Reason: {safe_reason}</div>" if safe_reason else ''}
            </div>
            ''', unsafe_allow_html=True)

    # Extracted Estimate Parts - Clear and Simple Section
    estimate_parts_display = st.session_state.get('extracted_estimate_parts', [])
    if estimate_parts_display:
        st.markdown('<h3 class="section-header">📋 Extracted Estimate Parts (From Estimate Copy)</h3>', unsafe_allow_html=True)
        st.markdown("**Parts found in the estimate document:**")
        
        # Display in a simple, clean format
        cols = st.columns(3)
        for i, ep in enumerate(estimate_parts_display):
            with cols[i % 3]:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #007bff;'>
                    <strong>🔧 {str(ep)}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"**Total parts found:** {len(estimate_parts_display)}")
        st.markdown("---")  # Add separator
    else:
        st.markdown('<h3 class="section-header">📋 Extracted Estimate Parts</h3>', unsafe_allow_html=True)
        st.info("No estimate copy was uploaded or no parts were found in the estimate document.")
    
    # Summary and recommendation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Summary:** {result['summary']}")
    with col2:
        recommendation_color = {
            "Approve": "🟢",
            "Investigate": "🟡",  
            "Deny": "🔴"
        }.get(result['recommendation'], "⚪")
        st.markdown(f"**Recommendation:** {recommendation_color} {result['recommendation']}")
    
    # Relevant damages
    if result.get('relevant_damages'):
        st.markdown('<h3 class="section-header">✅ Matched Damages</h3>', unsafe_allow_html=True)
        for damage in result['relevant_damages']:
            part = damage.get('part', '').title()
            desc = damage.get('damage_description', '')
            severity = (damage.get('severity', '') or '')
            # No severity display per request
            show_desc = (workflow != "with_dr")
            desc_html = f"<div style='margin-top:4px;'>{desc}</div>" if (show_desc and desc) else ''
            st.markdown(f'''
            <div class="damage-item" style="margin-bottom: 12px;">
                <span style="display: flex; align-items: center; justify-content: space-between;"><span><strong>{part}</strong></span></span>
                {desc_html}
            </div>
            ''', unsafe_allow_html=True)

    # Flagged/Doubtful damages (only those that could plausibly be damaged, but not 100% sure)
    if result.get('doubtful_damages'):
        plausible_flagged = []
        for damage in result['doubtful_damages']:
            # Only show if reason does NOT say impossible, but is flagged as doubtful
            reason = damage.get('reason', '').lower()
            if 'impossible' not in reason and 'not possible' not in reason and 'cannot' not in reason:
                plausible_flagged.append(damage)
        if plausible_flagged:
            st.markdown('<h3 class="section-header">❔ Doubtful Damages</h3>', unsafe_allow_html=True)
            for damage in plausible_flagged:
                st.markdown(f'''
                <div class="damage-item" style="margin-bottom: 12px;">
                    <div><strong>{damage.get('part', '').title()}</strong></div>
                    <div style='margin-top:4px;'>{damage.get('damage_description', '')}</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # Rejected parts
    if result.get('rejected_parts'):
        st.markdown('<h3 class="section-header">🚫 Rejected Parts</h3>', unsafe_allow_html=True)
        for rejected in result['rejected_parts']:
            st.markdown(f'''
            <div class="rejected-item">
                <div><strong>🚫 {rejected.get("part", "")}</strong></div>
            </div>
            ''', unsafe_allow_html=True)
    
    # No additional damages section
    
    # Download results
    st.markdown("---")
    st.markdown("### 📥 Download Final Report (PDF)")

    if 'pdf_data' not in st.session_state:
        st.session_state['pdf_data'] = None

    if workflow == "with_dr":
        render_pdf_download_link(st.session_state.get('pdf_data', b''), 'damage_report_matched_parts.pdf')
    else:
        render_pdf_download_link(st.session_state.get('pdf_data', b''), 'damage_report_matched_parts.pdf')

if __name__ == "__main__":
    if 'pdf_data' not in st.session_state:
        st.session_state['pdf_data'] = None
    main()
