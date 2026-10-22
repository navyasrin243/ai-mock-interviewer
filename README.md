# 🎯 AI Mock Interviewer

A resume-aware mock interview app. Upload your resume and paste a job description to get a personalized interview with **adaptive follow-up questions**, **voice answers**, and an **evidence-based report**.

## 🔗 Live Demo
https://ai-mock-interviewer-hsvhryyqhbsyd2ao8gl6dk.streamlit.app/

## ✨ Features
- **Personalized questions** from your resume and the target JD (concept, project and situational mix)
- **Adaptive follow-ups**: a weak answer (score below 5) triggers a targeted follow-up on the missing points, once per topic
- **Evidence-based scoring** on correctness, completeness and relevance (40/40/20). The final score is computed in Python, not by the LLM
- **Voice or text answers** (Groq Whisper transcription, editable before submit)
- **Final report**: overall score, per-topic scores, strong/weak topics, question-by-question feedback with a stronger sample answer, and a study plan
- **Prompt-injection guard** in the evaluator (candidate answers are treated as untrusted data)

## 🛠️ Tech Stack
| Area | Technology |
|------|------------|
| LLM | Groq (Qwen) via LangChain |
| Validation | Pydantic (strict types, Literal fields) |
| Voice | Groq Whisper (`whisper-large-v3-turbo`) + `st.audio_input` |
| UI / Deploy | Streamlit, Streamlit Community Cloud |
| Prototype | LangGraph state machine in `graph.py` (batch evaluation flow) |

## 🧱 Project Structure
```
app.py           Streamlit UI and live interview flow (incl. follow-ups)
question_gen.py  Resume + JD -> validated questions, topic extraction, follow-up generation
evaluator.py     Answer scoring, strict JSON validation, retries
voice.py         Speech-to-text via Groq Whisper
graph.py         LangGraph prototype of the interview flow
```

## 🚀 Run Locally
```bash
git clone https://github.com/navyasrin243/ai-mock-interviewer.git
cd ai-mock-interviewer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Then run:
```bash
streamlit run app.py
```

## ⚠️ Limitations
- Free-tier API limits can cause rate-limit errors under heavy use
- Scores come from an LLM judge, so treat them as practice feedback, not ground truth
- Scanned (image-only) PDF resumes are not supported
- Camera/body-language feedback is planned for a future version

## 👩‍💻 Author
**Navyasri Akula** · [GitHub](https://github.com/navyasrin243) · [LinkedIn](https://www.linkedin.com/in/navyasriakula/)
