import os, json, re
from typing import List, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field, field_validator

load_dotenv()

MIN_QUESTIONS = 3
MAX_QUESTIONS = 10
MAX_RETRIES   = 2


class Question(BaseModel):
    question:     str = Field(min_length=10)
    topic:        str = Field(min_length=3)
    difficulty:   Literal["easy", "medium", "hard"]
    q_type:       Literal["hr", "technical", "situational"]
    ideal_points: List[str]

    @field_validator("ideal_points")
    @classmethod
    def check_points(cls, v):
        cleaned = [p.strip() for p in v if p.strip()]
        if len(cleaned) < 2:
            raise ValueError("Need at least 2 ideal points")
        return cleaned


class QuestionSet(BaseModel):
    candidate_name: str
    target_role:    str
    key_topics:     List[str]
    questions:      List[Question]


def validate_inputs(resume_text, jd_text, interview_type, num_questions):
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if len(resume_text.strip()) < 50:
        raise ValueError("Resume too short.")
    if not isinstance(jd_text, str) or not jd_text.strip():
        raise ValueError("Job description is empty.")
    if interview_type not in ("hr", "technical", "mixed"):
        raise ValueError(f"Invalid interview_type: {interview_type}")
    if type(num_questions) is not int:
        raise ValueError("num_questions must be integer.")
    min_q = MIN_QUESTIONS if interview_type == "mixed" else 1
    if not min_q <= num_questions <= MAX_QUESTIONS:
        raise ValueError(
            f"num_questions must be {min_q}-{MAX_QUESTIONS} for {interview_type}."
        )


def load_resume(pdf_path: str) -> str:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Resume not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Must be a PDF file.")
    try:
        pages = PyPDFLoader(pdf_path).load()
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")
    text = "\n".join(p.page_content or "" for p in pages).strip()
    if not text:
        raise ValueError("No text found in PDF.")
    return text


def extract_resume_sections(resume_text: str, max_chars: int = 3000) -> str:
    text = resume_text.strip()
    if len(text) <= max_chars:
        return text
    priority = ("skills","technical skills","projects","experience",
                "internship","publications","certifications","summary","profile")
    skip     = ("education","references","hobbies","languages","awards")

    def is_heading(line, keywords):
        n = re.sub(r"[^a-z ]", " ", line.lower())
        n = " ".join(n.split())
        return any(n == k or n == k + " section" for k in keywords)

    kept, chars, in_priority = [], 0, False
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if is_heading(s, skip):
            in_priority = False
            continue
        if is_heading(s, priority):
            in_priority = True
            if chars + len(line) + 1 <= max_chars:
                kept.append(line)
                chars += len(line) + 1
            continue
        if in_priority:
            remaining = max_chars - chars
            if remaining <= 0: break
            kept.append(line[:remaining])
            chars += min(len(line), remaining) + 1
            if chars >= max_chars: break

    result = "\n".join(kept).strip()
    return result if len(result) >= 200 else text[:max_chars]


def get_type_distribution(interview_type: str, num_questions: int) -> dict:
    if interview_type == "hr":
        return {"hr": num_questions, "technical": 0, "situational": 0}
    if interview_type == "technical":
        return {"hr": 0, "technical": num_questions, "situational": 0}
    if num_questions == 3:
        return {"hr": 1, "technical": 1, "situational": 1}
    if num_questions == 4:
        return {"hr": 1, "technical": 2, "situational": 1}
    return {"hr": 1, "technical": num_questions - 2, "situational": 1}


def parse_json(response: str) -> dict:
    text = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"No JSON found: {response[:200]}")
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def parse_json_array(response: str) -> list:
    text = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end <= start:
        return []
    try:
        return json.loads(text[start:end])
    except:
        return []


def verify_output(result, num_questions, interview_type):
    issues = []
    if len(result.questions) != num_questions:
        issues.append(f"Expected {num_questions}, got {len(result.questions)}")
    if interview_type == "mixed":
        expected = get_type_distribution(interview_type, num_questions)
        actual   = {"hr": 0, "technical": 0, "situational": 0}
        for q in result.questions:
            actual[q.q_type] += 1
        for qt, exp in expected.items():
            if actual[qt] != exp:
                issues.append(f"Expected {exp} {qt}, got {actual[qt]}")
    return issues


def extract_topics_from_resume(resume_text: str, llm) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract technical skills from resume as interview topics.
Return SKILL names only — NOT project names.
Good: ["Python", "SQL", "Machine Learning", "LangChain"]
Bad:  ["AI Resume Analyzer", "Apple Leaf Detection"]
Return ONLY JSON array. Max 8 items. No markdown."""),
        ("human", "Resume:\n{resume}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"resume": resume_text[:2000]})
        topics   = parse_json_array(response)
        return [t for t in topics if isinstance(t, str)][:8]
    except Exception as e:
        print(f"Topic extraction failed: {e}")
        return ["Python", "Machine Learning", "SQL"]


def generate_followup_question(
    original_question, user_answer, missing_points,
    topic, resume_text, llm=None
) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = ChatGroq(model="qwen/qwen3.8-27b",
                       groq_api_key=api_key, temperature=0.5)
    missing_str = "\n".join(f"- {p}" for p in missing_points[:3])
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior MNC interviewer.
Generate ONE follow-up question targeting what was missing about: {topic}
Return ONLY JSON:
{{{{
  "question": "follow-up question",
  "topic": "{topic}",
  "difficulty": "medium",
  "q_type": "technical",
  "ideal_points": ["point1", "point2"]
}}}}"""),
        ("human", """Original: {original}
Answer: {answer}
Missing:
{missing}
JSON only:""")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "original": original_question,
            "answer":   user_answer[:500],
            "missing":  missing_str
        })
        data = parse_json(response)
        return {
            "question":     data.get("question",
                f"Elaborate on {topic} covering {missing_points[0] if missing_points else 'key concepts'}?"),
            "topic":        topic,
            "difficulty":   "medium",
            "q_type":       data.get("q_type", "technical"),
            "ideal_points": data.get("ideal_points",
                missing_points[:3] if missing_points else ["Explain clearly"]),
            "is_followup":  True
        }
    except Exception as e:
        print(f"Follow-up gen failed: {e}")
        mp = missing_points[0] if missing_points else "key concepts"
        return {
            "question":     f"Can you explain {topic} in more detail, covering {mp}?",
            "topic":        topic,
            "difficulty":   "medium",
            "q_type":       "technical",
            "ideal_points": missing_points[:3] if missing_points else ["Explain clearly"],
            "is_followup":  True
        }


def generate_questions(
    resume_text, jd_text,
    interview_type="mixed", num_questions=5
) -> QuestionSet:

    validate_inputs(resume_text, jd_text, interview_type, num_questions)
    resume_section = extract_resume_sections(resume_text)
    jd_section     = jd_text.strip()[:1500]
    dist           = get_type_distribution(interview_type, num_questions)

    if interview_type == "hr":
        type_instruction = (
            f"Generate exactly {num_questions} HR behavioral questions. "
            'All q_type must be "hr".'
        )
    elif interview_type == "technical":
        type_instruction = (
            f"Generate exactly {num_questions} technical questions. "
            'All q_type must be "technical". '
            "Mix concept questions AND project-based questions."
        )
    else:
        type_instruction = (
            f"Generate exactly {num_questions} questions:\n"
            f'- {dist["hr"]} with q_type "hr"\n'
            f'- {dist["technical"]} with q_type "technical"\n'
            f'- {dist["situational"]} with q_type "situational"\n'
            "Distribution must match exactly."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing from .env")

    llm = ChatGroq(model="qwen/qwen3.8-27b",
                   groq_api_key=api_key, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior MNC interviewer.
Generate BALANCED interview questions:

1. CONCEPT questions (40%): Core skill concepts
   "What is a Python decorator?"
   "Explain SQL JOIN types"
   "What is overfitting?"

2. PROJECT questions (40%): Based on candidate's ACTUAL projects only
   Never invent experience!

3. SCENARIO questions (20%): Real-world problems
   "Your model drops in production — what do you do?"

RULES:
- Cover MULTIPLE skills from resume
- Never ask only about projects
- Test depth of knowledge

{type_instruction}

Return ONLY valid JSON:
{{
  "candidate_name": "name from resume or Unknown",
  "target_role": "role from JD",
  "key_topics": ["skill1", "skill2"],
  "questions": [
    {{
      "question": "question text",
      "topic": "Python",
      "difficulty": "easy|medium|hard",
      "q_type": "hr|technical|situational",
      "ideal_points": ["point1", "point2", "point3"]
    }}
  ]
}}
No markdown. No text outside JSON."""),
        ("human", """Resume:
{resume}

Job Description:
{jd}

{retry_note}

Generate exactly {num_q} questions. JSON only:"""),
    ])

    chain      = prompt | llm | StrOutputParser()
    retry_note = ""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Generating: attempt {attempt}/{MAX_RETRIES}")
            response = chain.invoke({
                "type_instruction": type_instruction,
                "resume":           resume_section,
                "jd":               jd_section,
                "num_q":            num_questions,
                "retry_note":       retry_note,
            })
            data   = parse_json(response)
            result = QuestionSet.model_validate(data)
            issues = verify_output(result, num_questions, interview_type)
            if not issues:
                print(f"✅ Generated {len(result.questions)} questions")
                return result
            last_error = "; ".join(issues)
            retry_note = "Fix these issues:\n" + "\n".join(f"- {i}" for i in issues)
        except Exception as e:
            err = str(e)
            if "401" in err or "invalid_api_key" in err.lower():
                raise RuntimeError("Invalid Groq API key.")
            if "429" in err: raise RuntimeError("Rate limit.")
            if "503" in err: raise RuntimeError("Groq unavailable.")
            last_error = err
            retry_note = f"Previous failed: {err[:200]}. Return valid JSON."
        if attempt < MAX_RETRIES:
            print(f"Retrying: {last_error}")
        else:
            raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")
    raise RuntimeError("Question generation failed.")