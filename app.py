import streamlit as st
import json
import tempfile
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import requests
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
    body, .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0d47a1;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #b0c4de33;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #263238;
        margin-top: 2.2rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #1976d2;
        padding-left: 12px;
        background: linear-gradient(90deg, #e3f2fd 60%, #fff 100%);
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px #b0c4de22;
    }
    .result-box {
        background: linear-gradient(90deg, #f8fafc 60%, #e3f2fd 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 0.7rem;
        border-left: 5px solid #1976d2;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px #b0c4de22;
    }
    .damage-item {
        background: linear-gradient(90deg, #e3f2fd 60%, #f5f7fa 100%);
        padding: 0.7rem 1rem;
        border-radius: 0.5rem;
        margin: 0.6rem 0;
        border-left: 4px solid #1976d2;
        color: #263238;
        font-weight: 600;
        box-shadow: 0 1px 4px #b0c4de22;
        transition: box-shadow 0.2s;
    }
    .damage-item:hover {
        box-shadow: 0 4px 16px #1976d233;
    }
    .missing-item {
        background: linear-gradient(90deg, #fffde7 60%, #fff 100%);
        padding: 0.7rem 1rem;
        border-radius: 0.5rem;
        margin: 0.6rem 0;
        border-left: 4px solid #ffc107;
        color: #b28704;
        font-weight: 600;
        box-shadow: 0 1px 4px #ffe08233;
    }
    .additional-item {
        background: linear-gradient(90deg, #ffebee 60%, #fff 100%);
        padding: 0.7rem 1rem;
        border-radius: 0.5rem;
        margin: 0.6rem 0;
        border-left: 4px solid #d32f2f;
        color: #b71c1c;
        font-weight: 600;
        box-shadow: 0 1px 4px #ffcdd233;
    }
    .rejected-item {
        background: linear-gradient(90deg, #fce4ec 60%, #fff 100%);
        padding: 0.7rem 1rem;
        border-radius: 0.5rem;
        margin: 0.6rem 0;
        border-left: 4px solid #e91e63;
        color: #ad1457;
        font-weight: 600;
        box-shadow: 0 1px 4px #f8bbd933;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 260px;
        background: #1976d2;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 10px 14px;
        position: absolute;
        z-index: 2;
        top: -5px;
        left: 110%;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 1em;
        box-shadow: 0 2px 8px #1976d233;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .info-icon {
        color: #1976d2;
        margin-left: 8px;
        font-size: 1.2em;
        vertical-align: middle;
        filter: drop-shadow(0 1px 2px #b0c4de66);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1976d2 60%, #42a5f5 100%);
        color: #fff;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px #1976d233;
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565c0 60%, #1976d2 100%);
        box-shadow: 0 4px 16px #1976d233;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #388e3c 60%, #66bb6a 100%);
        color: #fff;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px #388e3c33;
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #2e7d32 60%, #388e3c 100%);
        box-shadow: 0 4px 16px #388e3c33;
    }
    .stExpanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1976d2 !important;
    }
    .stTextArea textarea {
        background: #f5f7fa !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        color: #263238 !important;
        border: 1.5px solid #b0bec5 !important;
    }
    .stAlert {
        border-radius: 8px !important;
        font-size: 1.05rem !important;
    }
    .stImage>img {
        border-radius: 12px !important;
        box-shadow: 0 2px 12px #b0c4de33;
    }
</style>
""", unsafe_allow_html=True)

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
    return result['parts']

def extract_parts_from_estimate_copy(estimate_text: str, model_name: str = "gpt-4o") -> list:
    """Extract car parts names from estimate copy text"""
    if not estimate_text or estimate_text.strip() == "":
        return []
    
    prompt = f"""
    You are an expert automotive parts specialist. Analyze the following estimate copy text and extract ALL vehicle part names mentioned in it.
    
    Focus on identifying:
    - Body parts (bumpers, doors, fenders, hoods, trunks, etc.)
    - Lights (headlights, tail lights, fog lights, etc.)
    - Glass components (windshields, windows, mirrors, etc.)
    - Mechanical parts (engine components, suspension, brakes, etc.)
    - Interior parts (seats, dashboard, airbags, etc.)
    - Electrical components (wiring, sensors, etc.)
    - Structural parts (frame, pillars, panels, etc.)
    
    Extract only the actual vehicle part names, not prices, labor, or other non-part items.
    
    ESTIMATE COPY TEXT:
    {estimate_text}
    
    Return ONLY a JSON array of part names, e.g. ["front bumper", "headlights", "windshield", "radiator", "door panel"]
    If no parts are found, return an empty array [].
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an expert automotive parts specialist."},
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

def filter_damage_report(damage_report_text: str, potential_parts: list, model_name: str = "gpt-4o", estimate_text: str = "", estimate_parts: list = None) -> dict:
    """Filter damage report to only show damages related to potential parts from story"""
    
    # Use provided estimate_parts or fallback to empty list
    if estimate_parts is None:
        estimate_parts = []
    
    prompt = f"""
    You are an expert insurance claims agent. You have received 4 car images (front, back, left, right) and a claim form story describing the accident. Your job is to analyze the detected damaged parts from all images and:

    1. Identify ONLY those parts that you believe are damaged as a direct result of the same accident described in the claim form story. These should be listed as 'relevant_damages'.
    2. Flag any other parts that are damaged but you are doubtful about their connection to the described accident, but which could plausibly be affected due to a collision. These should be listed as 'doubtful_damages'.
    3. Remove and ignore any parts that are impossible to be affected by the described accident scenario, based on physical plausibility. For example, if the story says the car was hit from the front, then rear bumper damage is impossible and must be excluded, with a reason.
    4. For each part, provide a clear and concise 'reason' for its classification, including why it is possible or impossible given the accident scenario.
    5. For rejected parts analysis, analyze ONLY the parts from the extracted estimate parts list below and reject them if they meet either criteria:
       a) Parts that show NO damage according to the damage report images from any view (front, back, left, right)
       b) Parts that might show damage in the images but CANNOT be plausibly linked to this specific accident based on the claim story (e.g., damage that appears to be from a previous accident, different impact direction, or unrelated incident)
       IMPORTANT: Only consider parts from the extracted estimate parts list. Do not include any parts from the claim story or generate any parts yourself. Use only the parts provided in the list below.
    6. For doubtful damages, analyze internal/hidden parts that could be damaged based on the accident scenario but are not visible in the images. Select 6-7 relevant parts from this comprehensive list based on collision type:
       • Front impact parts: Radiator, condenser, engine mounts, cooling fan, intercooler, timing components, front crossmember, transmission cooler, wiring harness, air intake system
       • Rear impact parts: Rear panel, boot floor, spare wheel housing, rear impact bar, rear crash sensors, fuel tank, wiring harness, rear suspension links, trunk latch mechanism
       • Side impact parts: Door beams, curtain airbags, seatbelt pretensioners, wiring inside doors, window regulator, power window motor, side impact sensors, B-pillar, seats and frames
       • Hidden/Internal damage: Frame rails, subframe, suspension mounts, steering column (front), rear axle beam (rear), floor pan, insulation, soundproofing materials
    7. Provide a brief summary and a recommendation (Approve/Investigate/Deny) based on your findings.

    POTENTIAL PARTS FROM STORY: {potential_parts}
    DETECTED DAMAGED PARTS FROM IMAGES:
    {damage_report_text}
    EXTRACTED PARTS FROM ESTIMATE COPY: {estimate_parts}
    ESTIMATE COPY: {estimate_text}

    IMPORTANT FOR REJECTED PARTS: Use only the parts from the "EXTRACTED PARTS FROM ESTIMATE COPY" list above. Evaluate each part in that list against the damage report images to determine if it should be rejected based on the two criteria mentioned.

    Return your analysis in this JSON format:
    {{
        "relevant_damages": [
            {{
                "part": "front bumper",
                "damage_description": "Front bumper dented and scratched",
                "severity": "medium",
                "found_in_report": true,
                "reason": "Directly matches the accident scenario described in the claim form."
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
    - Only include parts in 'relevant_damages' if you are confident they are damaged due to the same accident as described in the claim form story.
    - For 'doubtful_damages', select 6-7 relevant internal/hidden parts from the provided lists based on collision type (front/rear/side impact). Choose parts that could plausibly be affected by the impact forces but are not visible in images.
    - Do NOT include any parts that are impossible to be affected by the described accident scenario (e.g., rear impact parts in a front collision). Always provide a reason for exclusion.
    - For each part, provide a clear 'reason' for its inclusion or exclusion, referencing the physical plausibility and accident details.
    - Be strict and accurate in your analysis. Never include impossible damages.
    - 'rejected_parts' should ONLY include parts from the "EXTRACTED PARTS FROM ESTIMATE COPY" list that meet either criteria: (a) NOT found damaged in the damage report images from any view, OR (b) show damage but cannot be plausibly linked to this specific accident based on the claim story. Use only the parts provided in the extracted list. Do not include any parts from claim story or generate parts yourself.
    - For doubtful damages, focus on selecting the most relevant internal components from the provided lists that match the collision type and impact direction described in the claim story.
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
        "Side: Doors, windows, door handles, side mirrors, side skirts, wheel arches, wheels, tires, fuel cap, turn signals, pillars (A, B, C). "
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

def generate_matching_parts_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=None, rejected_parts=None, ai_interpreted_parts=None, consistency_report=None):
    # Use Streamlit cache to avoid regenerating PDF on rerun
    import streamlit as st
    @st.cache_data(show_spinner=False)
    def _generate_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=None, rejected_parts=None, ai_interpreted_parts=None, consistency_report=None):
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
                severity = damage.get('severity', '')
                reason = damage.get('reason', '')
                elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
                elements.append(Paragraph(f"<b>Description:</b> {desc}", normal_style))
                if severity:
                    elements.append(Paragraph(f"<b>Severity:</b> {severity}", normal_style))
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
                elements.append(Paragraph(f"<b>Description:</b> {desc}", normal_style))
                elements.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
                elements.append(Spacer(1, 0.12*inch))
        
        # Rejected Parts section
        if rejected_parts:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Rejected Parts (Claimed but Not Found in Damage Report)</b>", heading_style))
            for rejected in rejected_parts:
                part = rejected.get('part', '')
                reason = rejected.get('reason', '')
                elements.append(Paragraph(f"<b>Part:</b> {part}", normal_style))
                elements.append(Paragraph(f"<b>Reason for Rejection:</b> {reason}", normal_style))
                elements.append(Spacer(1, 0.12*inch))
        
        # Consistency Check section
        if consistency_report:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Accident Consistency Analysis</b>", heading_style))
            elements.append(Paragraph(consistency_report, normal_style))
        
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf
    return _generate_pdf(policy_story, matching_parts, summary, recommendation, damage_report_file, model_used, doubtful_parts=doubtful_parts, rejected_parts=rejected_parts, ai_interpreted_parts=ai_interpreted_parts, consistency_report=consistency_report)

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

def main():
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 📋 Analysis Options")
        analysis_option = st.radio(
            "Choose your analysis method:",
            ["With Damage Report", "Without Damage Report"],
            index=1  # Default to "Without Damage Report Filter"
        )
    
    if analysis_option == "With Damage Report":
        with_damage_report_workflow()
    else:
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
                with st.expander("📋 Estimate PDF Preview"):
                    st.text_area("Extracted Estimate Content", estimate_text, height=200, disabled=True)
            else:
                estimate_text = uploaded_estimate.read().decode('utf-8')
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ Text estimate uploaded: {uploaded_estimate.name}")
                with st.expander("📋 Estimate Preview"):
                    st.text_area("Estimate Content", estimate_text, height=200, disabled=True)
        
        # Require estimate before proceeding
        if uploaded_estimate is not None:
            if st.button("Analyze Reports and Forms", type="primary", use_container_width=True):
                st.session_state['estimate_uploaded'] = True
                st.rerun()
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

    with st.spinner("🔍 Filtering damage report based on claim form..."):
        result = filter_damage_report(damage_report_text, potential_parts, model_name, estimate_text, estimate_parts)

    # Display results
    display_analysis_results(potential_parts, result, file_name, model_name)

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
                with st.expander("📋 Estimate PDF Preview"):
                    st.text_area("Extracted Estimate Content", estimate_text, height=200, disabled=True)
            else:
                estimate_text = uploaded_estimate.read().decode('utf-8')
                st.session_state['estimate_text'] = estimate_text
                st.session_state['estimate_file_name'] = uploaded_estimate.name
                st.success(f"✅ Text estimate uploaded: {uploaded_estimate.name}")
                with st.expander("📋 Estimate Preview"):
                    st.text_area("Estimate Content", estimate_text, height=200, disabled=True)
        
        # Require estimate before proceeding
        if uploaded_estimate is not None:
            if st.button("Analyze Images and Forms", type="primary", use_container_width=True):
                st.session_state['estimate_uploaded'] = True
                st.rerun()
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
        # Combine all detected parts into a single damage report text
        damage_report_text = "\n".join(all_detected_parts)
        with st.spinner("Inferring potential damages from description..."):
            potential_parts = extract_damage_parts_from_story(claim_description, model_name)
        with st.spinner("🔍 Extracting parts from estimate copy..."):
            estimate_parts = extract_parts_from_estimate_copy(estimate_text, model_name)
        with st.spinner("Filtering damage report based on description..."):
            result = filter_damage_report(damage_report_text, potential_parts, model_name, estimate_text, estimate_parts)
            # Remove additional damages from result
            if 'additional_damages' in result:
                result['additional_damages'] = []
            # Ensure rejected_parts field exists
            if 'rejected_parts' not in result:
                result['rejected_parts'] = []
            # Remove missing_parts if it exists (now combined with rejected_parts)
            if 'missing_parts' in result:
                del result['missing_parts']
        combined_image_names = ", ".join([name for name in image_names.values()])
        display_combined_analysis(per_image_potential_parts, potential_parts, result, combined_image_names, model_name)
def display_combined_analysis(per_image_potential_parts, potential_parts, result, file_name, model_name):
    """Display 4 sections for AI-Interpreted Potentially Impacted Parts, rest combined"""
    st.markdown('<div style="height: 24px"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header" style="margin-bottom: 2.5rem;">🔎 Analysis Results</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height: 16px"></div>', unsafe_allow_html=True)
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
                        <span style="display: flex; align-items: center; justify-content: space-between;">
                            <span>🔧 {part}</span>
                            <span class="tooltip">
                                <span class="info-icon">&#9432;</span>
                                <span class="tooltiptext">{reason}</span>
                            </span>
                        </span>
                    </div>
                    ''', unsafe_allow_html=True)
            st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

    # Summary and recommendation (combined)
    st.markdown('<div style="height: 24px"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"<div style='font-size:1.15rem; margin-bottom:10px;'><b>Summary:</b> {result['summary']}</div>", unsafe_allow_html=True)
    with col2:
        recommendation_color = {
            "Approve": "🟢",
            "Investigate": "🟡",
            "Deny": "🔴"
        }.get(result['recommendation'], "⚪")
        st.markdown(f"<div style='font-size:1.15rem; margin-bottom:10px;'><b>Recommendation:</b> {recommendation_color} {result['recommendation']}</div>", unsafe_allow_html=True)
    st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # Relevant damages (combined)
    if result.get('relevant_damages'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">✅ Matched Damages</h3>', unsafe_allow_html=True)
        for damage in result['relevant_damages']:
            severity_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(damage.get('severity', ''), "⚪")
            reason = damage.get('reason', '')
            st.markdown(f'''
            <div class="damage-item" style="margin-bottom: 18px;">
                <span style="display: flex; align-items: center; justify-content: space-between;">
                    <span><strong>{damage.get('part', '').title()}</strong></span>
                    <span class="tooltip">
                        <span class="info-icon">&#9432;</span>
                        <span class="tooltiptext">{reason}</span>
                    </span>
                </span>
                <div style='margin-top:4px;'>{damage.get('damage_description', '')}</div>
                <small style='margin-top:2px; display:block;'>Severity: {severity_color} {damage.get('severity', '').title()}</small>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # Doubtful damages (combined)
    if result.get('doubtful_damages'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">❔ Doubtful Damages</h3>', unsafe_allow_html=True)
        for damage in result['doubtful_damages']:
            reason = damage.get('reason', '')
            st.markdown(f'''
            <div class="damage-item" style="margin-bottom: 18px;">
                <span><strong>{damage.get('part', '').title()}</strong></span>
                <div style='margin-top:4px;'>{damage.get('damage_description', '')}</div>
                <small style='margin-top:6px; display:block; color:#666; font-style:italic;'>{reason}</small>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # Rejected parts (combined)
    if result.get('rejected_parts'):
        st.markdown('<h3 class="section-header" style="margin-bottom: 1.5rem;">🚫 Rejected Parts (Claimed but Not Found in Damage Report)</h3>', unsafe_allow_html=True)
        for rejected in result['rejected_parts']:
            reason = rejected.get('reason', '')
            st.markdown(f'''
            <div class="rejected-item" style="margin-bottom: 14px;">
                <span style="display: flex; align-items: center; justify-content: space-between;">
                    <span>🚫 {rejected["part"]}</span>
                    <span class="tooltip">
                        <span class="info-icon">&#9432;</span>
                        <span class="tooltiptext">{reason}</span>
                    </span>
                </span>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    # --- Consistency Check LLM Call ---
    st.markdown('<div style="height: 32px"></div>', unsafe_allow_html=True)
    image_damage_dict = {k: [p[0] for p in v] for k, v in per_image_potential_parts.items()}
    claim_form_story = st.session_state.get('claim_description', '') or st.session_state.get('policy_story', '')
    consistency_prompt = f"""
    You are an expert vehicle accident analyst. Analyze the damage distribution across multiple car images to validate consistency with a single accident event.

    Detected damaged parts from car images:
    {image_damage_dict}

    Accident description: {claim_form_story}

    Validate whether damages seen across multiple images align with a single accident:
    - If damage appears on left and right sides but not front/back, flag as potentially inconsistent (suggests multiple incidents)
    - Check if impact direction matches damage pattern (front collision should show front damage, not isolated rear damage)
    - Verify damage severity is consistent with described impact force and direction
    - Look for contradictory damage patterns (e.g., both front and rear damage from a single-direction collision)
    - Assess if damage distribution follows logical impact physics and force transfer

    Provide a brief 3-4 line consistency assessment focusing on whether the damage pattern supports a single accident scenario.
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
    st.markdown('<div style="height: 32px"></div>', unsafe_allow_html=True)
    # Download results (combined)
    st.markdown("---")
    st.markdown('<div style="height: 18px"></div>', unsafe_allow_html=True)
    st.markdown("### 📥 Download Final Report (PDF)")
    st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)
    matched_parts = result.get('relevant_damages', [])
    doubtful_parts = result.get('doubtful_damages', [])
    rejected_parts = result.get('rejected_parts', [])
    pdf_bytes = generate_matching_parts_pdf(
        policy_story="",  # This will be empty for image-based analysis
        matching_parts=matched_parts,
        summary=result.get('summary', ''),
        recommendation=result.get('recommendation', ''),
        damage_report_file=file_name,
        model_used=model_name,
        doubtful_parts=doubtful_parts,
        rejected_parts=rejected_parts,
        ai_interpreted_parts=per_image_potential_parts,
        consistency_report=consistency_report
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="damage_report_matched_parts.pdf",
        mime="application/pdf"
    )

def display_analysis_results(potential_parts, result, file_name, model_name):
    """Display the analysis results for both workflows"""
    st.markdown('<h2 class="section-header">🔎 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Potential parts section
    st.markdown('<h3 class="section-header">🤖 AI-Interpreted Potentially Impacted Parts</h3>', unsafe_allow_html=True)
    parts_cols = st.columns(3)
    for i, part_tuple in enumerate(potential_parts):
        part = part_tuple[0]
        reason = part_tuple[1] if len(part_tuple) > 1 else ""
        with parts_cols[i % 3]:
            st.markdown(f'''
            <div class="damage-item">
                <span style="display: flex; align-items: center; justify-content: space-between;">
                    <span>🔧 {part}</span>
                    <span class="tooltip">
                        <span class="info-icon">&#9432;</span>
                        <span class="tooltiptext">{reason}</span>
                    </span>
                </span>
            </div>
            ''', unsafe_allow_html=True)
    
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
            severity_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(damage.get('severity', ''), "⚪")
            reason = damage.get('reason', '')
            st.markdown(f'''
            <div class="damage-item">
                <span style="display: flex; align-items: center; justify-content: space-between;">
                    <span><strong>{damage.get('part', '').title()}</strong></span>
                    <span class="tooltip">
                        <span class="info-icon">&#9432;</span>
                        <span class="tooltiptext">{reason}</span>
                    </span>
                </span>
                {damage.get('damage_description', '')}<br>
                <small>Severity: {severity_color} {damage.get('severity', '').title()}</small>
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
            st.markdown('<h3 class="section-header">❔ Flagged Parts</h3>', unsafe_allow_html=True)
            for damage in plausible_flagged:
                reason = damage.get('reason', '')
                st.markdown(f'''
                <div class="damage-item">
                    <span><strong>{damage.get('part', '').title()}</strong></span>
                    <div style='margin-top:4px;'>{damage.get('damage_description', '')}</div>
                    <small style='margin-top:6px; display:block; color:#666; font-style:italic;'>{reason}</small>
                </div>
                ''', unsafe_allow_html=True)
    
    # Rejected parts
    if result.get('rejected_parts'):
        st.markdown('<h3 class="section-header">🚫 Rejected Parts (Claimed but Not Found in Damage Report)</h3>', unsafe_allow_html=True)
        for rejected in result['rejected_parts']:
            reason = rejected.get('reason', '')
            st.markdown(f'''
            <div class="rejected-item">
                <span style="display: flex; align-items: center; justify-content: space-between;">
                    <span>🚫 {rejected["part"]}</span>
                    <span class="tooltip">
                        <span class="info-icon">&#9432;</span>
                        <span class="tooltiptext">{reason}</span>
                    </span>
                </span>
            </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # No additional damages section
    
    # Download results
    st.markdown("---")
    st.markdown("### 📥 Download Final Report (PDF)")
    matched_parts = result.get('relevant_damages', [])
    doubtful_parts = result.get('doubtful_damages', [])
    rejected_parts = result.get('rejected_parts', [])
    # Convert potential_parts to AI-interpreted format for PDF
    ai_interpreted_parts = {"claim_story": potential_parts} if potential_parts else None
    pdf_bytes = generate_matching_parts_pdf(
        policy_story="",  # This will be empty for image-based analysis
        matching_parts=matched_parts,
        summary=result.get('summary', ''),
        recommendation=result.get('recommendation', ''),
        damage_report_file=file_name,
        model_used=model_name,
        doubtful_parts=doubtful_parts,
        rejected_parts=rejected_parts,
        ai_interpreted_parts=ai_interpreted_parts,
        consistency_report=None  # No consistency check for damage report workflow
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="damage_report_matched_parts.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()