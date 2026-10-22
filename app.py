import copy
import hashlib
import html
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="AI Mock Interviewer", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
.main-title{font-size:2rem;font-weight:bold;text-align:center;color:#1f77b4;}
.question-box{background:#f0f7ff;color:#0b2540;padding:20px;border-radius:10px;
  border-left:5px solid #1f77b4;font-size:1.05rem;margin:10px 0;line-height:1.6;}
.fb-good{background:#d4edda;color:#155724;padding:10px;border-radius:8px;
  margin:4px 0;border-left:4px solid #27ae60;}
.fb-bad{background:#f8d7da;color:#721c24;padding:10px;border-radius:8px;
  margin:4px 0;border-left:4px solid #e74c3c;}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {
    "stage": "setup",
    "questions": [],
    "current_q": 0,
    "evaluations": [],
    "resume_text": "",
    "jd_text": "",
    "prev_eval": None,
    "topics": [],
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = copy.deepcopy(_v)


# ── Helpers ───────────────────────────────────────────────────────
def render_points(points, css_class):
    for p in points or []:
        if isinstance(p, str) and p.strip():
            st.markdown(
                f'<div class="{css_class}">• {html.escape(p)}</div>',
                unsafe_allow_html=True,
            )


def compute_report(evals):
    done = [
        e for e in evals
        if e.get("status") == "evaluated" and e.get("final_score") is not None
    ]
    failed = [e for e in evals if e.get("status") == "failed"]
    scores = [e["final_score"] for e in done]
    overall = round(sum(scores) / len(scores), 1) if scores else None

    if overall is None:
        level = "Unavailable"
    elif overall >= 8:
        level = "Excellent 🌟"
    elif overall >= 6:
        level = "Good ✅"
    elif overall >= 4:
        level = "Average ⚠️"
    else:
        level = "Needs Improvement ❌"

    by_topic = {}
    for e in done:
        by_topic.setdefault(e["topic"], []).append(e["final_score"])
    topic_scores = {t: round(sum(s) / len(s), 1) for t, s in by_topic.items()}

    return {
        "overall": overall,
        "level": level,
        "asked": len(evals),
        "evaluated": len(done),
        "failed": len(failed),
        "topic_scores": topic_scores,
        "weak": [t for t, s in topic_scores.items() if s <= 4],
        "strong": [t for t, s in topic_scores.items() if s >= 8],
    }


st.markdown('<p class="main-title">🎯 AI Mock Interviewer</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# STAGE 1 — SETUP
# ════════════════════════════════════════════════════════════════
if st.session_state.stage == "setup":
    st.markdown("#### Upload resume + paste JD → personalized adaptive interview!")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader("PDF only", type=["pdf"])
    with col2:
        st.subheader("💼 Job Description")
        jd_text = st.text_area(
            "Paste JD here", height=180,
            placeholder="We are looking for an AI Engineer...",
        )

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        interview_type = st.selectbox("Interview Type", ["mixed", "technical", "hr"])
    with col4:
        min_q = 3 if interview_type == "mixed" else 1
        num_questions = st.slider("Number of Questions", min_value=min_q, max_value=10, value=5)

    st.info(
        "💡 Questions are based on YOUR resume skills. "
        "If your answer is weak (score below 5), a follow-up question comes immediately."
    )
    st.divider()

    if st.button("🚀 Start Interview", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("❌ Upload your resume PDF!")
        elif not jd_text.strip():
            st.error("❌ Paste the Job Description!")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                from question_gen import generate_questions, load_resume

                with st.spinner("📄 Loading resume..."):
                    resume_text = load_resume(tmp_path)

                with st.spinner("🤖 Generating questions... (30-45 sec)"):
                    q_set = generate_questions(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        interview_type=interview_type,
                        num_questions=num_questions,
                    )

                st.session_state.resume_text = resume_text
                st.session_state.jd_text = jd_text
                st.session_state.topics = list(q_set.key_topics)
                st.session_state.questions = [
                    {
                        "question": q.question,
                        "topic": q.topic,
                        "difficulty": q.difficulty,
                        "q_type": q.q_type,
                        "ideal_points": q.ideal_points,
                        "is_followup": False,
                    }
                    for q in q_set.questions
                ]
                st.session_state.evaluations = []
                st.session_state.current_q = 0
                st.session_state.prev_eval = None
                st.session_state.stage = "interview"
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ════════════════════════════════════════════════════════════════
# STAGE 2 — INTERVIEW
# ════════════════════════════════════════════════════════════════
elif st.session_state.stage == "interview":
    questions = st.session_state.questions
    idx = st.session_state.current_q
    total = len(questions)

    if idx >= total:
        st.session_state.stage = "report"
        st.rerun()

    q = questions[idx]
    is_followup = q.get("is_followup", False)

    st.progress(idx / total, text=f"Question {idx + 1} of {total}")

    if st.session_state.topics:
        badges = " ".join(
            '<span style="background:#e3f2fd;color:#1565c0;padding:2px 8px;'
            f'border-radius:10px;font-size:0.8rem">{html.escape(str(t))}</span>'
            for t in st.session_state.topics
        )
        st.markdown(f"**Skills:** {badges}", unsafe_allow_html=True)
    st.divider()

    # Feedback on the previous answer
    prev = st.session_state.prev_eval
    if prev:
        score = prev["final_score"]
        with st.expander(f"📝 Feedback on your previous answer — {score}/10", expanded=True):
            if score >= 8:
                st.success(f"🌟 Excellent! {score}/10")
            elif score >= 6:
                st.success(f"✅ Good! {score}/10")
            elif score >= 4:
                st.warning(f"⚠️ Average. {score}/10")
            else:
                st.error(f"❌ Needs work. {score}/10")

            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**✅ Well done:**")
                render_points(prev.get("good_points"), "fb-good")
            with fc2:
                st.markdown("**❌ Missing:**")
                render_points(prev.get("missing_points"), "fb-bad")
            if score < 7 and prev.get("better_answer"):
                st.markdown("**💡 Stronger answer:**")
                st.info(prev["better_answer"])

    if is_followup:
        st.info("🔄 Follow-up — your previous answer was missing some key points.")

    label = "(Follow-up) " if is_followup else ""
    st.markdown(
        f'<div class="question-box"><b>Q{idx + 1} {label}'
        f'[{q["q_type"].upper()} | {q["difficulty"]}]</b><br><br>'
        f'{html.escape(q["question"])}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"📌 Topic: {q['topic']}")
    st.divider()

    # ── Answer input: typing or voice ────────────────────────────
    mode = st.radio(
        "Answer method", ["⌨️ Type", "🎤 Voice"],
        horizontal=True, key=f"mode_{idx}",
    )
    answer = ""

    if mode == "⌨️ Type":
        answer = st.text_area(
            "Your answer", height=150, key=f"text_{idx}",
            placeholder="Be specific. Use examples from your experience...",
        )
    else:
        st.markdown("🎤 **Record your answer** (allow microphone access in the browser)")
        audio = st.audio_input("Click the mic to record", key=f"audio_{idx}")

        if audio is not None:
            data = audio.getvalue()
            digest = hashlib.md5(data).hexdigest()[:10]
            cache_key = f"tr_{idx}_{digest}"

            if cache_key not in st.session_state:
                with st.spinner("🎤 Transcribing..."):
                    from voice import transcribe_audio
                    st.session_state[cache_key] = transcribe_audio(data)

            transcript = st.session_state[cache_key]
            if transcript:
                answer = st.text_area(
                    "Transcript (edit if needed):",
                    value=transcript, height=120, key=f"ve_{idx}_{digest}",
                )
            else:
                st.warning("Could not transcribe. Please type your answer below.")
                answer = st.text_area("Type your answer:", height=120, key=f"vfb_{idx}")

    bc1, bc2 = st.columns([4, 1])
    submit = bc1.button("Submit →", type="primary", use_container_width=True, key=f"sub_{idx}")
    skip = bc2.button("Skip ⏭️", use_container_width=True, key=f"skip_{idx}")

    if submit or skip:
        final_answer = answer.strip() if submit else ""

        if submit and len(final_answer) < 10:
            st.warning("Please write a more complete answer, or press Skip.")
            st.stop()

        record = {
            "question": q["question"],
            "topic": q["topic"],
            "q_type": q["q_type"],
            "difficulty": q["difficulty"],
            "is_followup": is_followup,
            "user_answer": final_answer,
        }
        feedback = None

        with st.spinner("🤖 Evaluating your answer..."):
            try:
                from evaluator import evaluate_answer

                r = evaluate_answer(
                    question=q["question"],
                    user_answer=final_answer,
                    ideal_points=q["ideal_points"],
                    q_type=q["q_type"],
                    resume_context=st.session_state.resume_text[:200],
                )
                feedback = {
                    "final_score": r.final_score,
                    "correctness": r.correctness,
                    "completeness": r.completeness,
                    "relevance": r.relevance,
                    "good_points": r.good_points,
                    "missing_points": r.missing_points,
                    "better_answer": r.better_answer,
                }
                record.update(feedback)
                record["status"] = "evaluated"
            except Exception as e:
                record.update({"final_score": None, "status": "failed", "error": str(e)})

            st.session_state.evaluations.append(record)
            st.session_state.prev_eval = feedback

            # Dynamic follow-up: weak answer, only once per topic
            if submit and feedback and feedback["final_score"] < 5 and not is_followup:
                try:
                    from question_gen import generate_followup_question

                    fq = generate_followup_question(
                        original_question=q["question"],
                        user_answer=final_answer,
                        missing_points=feedback["missing_points"],
                        topic=q["topic"],
                        resume_text=st.session_state.resume_text,
                    )
                    fq["is_followup"] = True
                    if fq.get("q_type") not in ("hr", "technical", "situational"):
                        fq["q_type"] = "technical"
                    if fq.get("difficulty") not in ("easy", "medium", "hard"):
                        fq["difficulty"] = "medium"
                    pts = fq.get("ideal_points")
                    if not isinstance(pts, list) or not pts:
                        fq["ideal_points"] = feedback["missing_points"][:3] or ["Explain clearly"]
                    st.session_state.questions.insert(idx + 1, fq)
                except Exception as e:
                    print(f"Follow-up generation failed: {e}")

        st.session_state.current_q += 1
        st.rerun()


# ════════════════════════════════════════════════════════════════
# STAGE 3 — REPORT
# ════════════════════════════════════════════════════════════════
elif st.session_state.stage == "report":
    evals = st.session_state.evaluations
    report = compute_report(evals)

    st.subheader("📊 Interview Report")
    st.divider()

    overall = report["overall"]
    if overall is not None:
        if overall >= 8:
            bg, fg = "#d4edda", "#155724"
        elif overall >= 6:
            bg, fg = "#d1ecf1", "#0c5460"
        elif overall >= 4:
            bg, fg = "#fff3cd", "#856404"
        else:
            bg, fg = "#f8d7da", "#721c24"
        st.markdown(
            f'<div style="background:{bg};color:{fg};font-size:2rem;font-weight:bold;'
            f'text-align:center;padding:18px;border-radius:10px;margin:10px 0;">'
            f'Overall Score: {overall}/10 — {report["level"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Could not calculate a score (no successful evaluations).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Asked", report["asked"])
    m2.metric("Evaluated", report["evaluated"])
    m3.metric("Failed", report["failed"])

    if report["topic_scores"]:
        st.divider()
        st.subheader("📈 Performance by Topic")
        cols = st.columns(min(len(report["topic_scores"]), 4))
        for i, (t, s) in enumerate(report["topic_scores"].items()):
            cols[i % len(cols)].metric(t, f"{s}/10")

    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("### 💪 Strong Topics")
        for t in report["strong"]:
            st.success(f"✅ {t}")
        if not report["strong"]:
            st.info("Keep practicing!")
    with sc2:
        st.markdown("### 📚 Weak Topics")
        for t in report["weak"]:
            st.error(f"❌ {t}")
        if not report["weak"]:
            st.success("No weak topics! 🎉")

    st.divider()
    st.subheader("📋 Question-by-Question Review")
    for i, e in enumerate(evals, 1):
        icon = "✅" if e.get("status") == "evaluated" else "❌"
        fu = " 🔄 Follow-up" if e.get("is_followup") else ""
        title = f"Q{i}{fu} {icon}: {e['question'][:65]}..."
        with st.expander(title):
            st.markdown(
                f"**Topic:** {e.get('topic', '')} | "
                f"**Type:** {e.get('q_type', '').upper()} | "
                f"**Difficulty:** {e.get('difficulty', '')}"
            )
            if e.get("status") == "evaluated":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score", f"{e['final_score']}/10")
                c2.metric("Correctness", f"{e.get('correctness', 0)}/10")
                c3.metric("Completeness", f"{e.get('completeness', 0)}/10")
                c4.metric("Relevance", f"{e.get('relevance', 0)}/10")

                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**✅ Well done:**")
                    render_points(e.get("good_points"), "fb-good")
                with g2:
                    st.markdown("**❌ Missing:**")
                    render_points(e.get("missing_points"), "fb-bad")

                if e.get("better_answer"):
                    st.markdown("**💡 Stronger answer:**")
                    st.info(e["better_answer"])
                if e.get("user_answer"):
                    st.markdown("**📝 Your answer:**")
                    st.write(e["user_answer"])
            else:
                st.warning(f"Evaluation failed: {e.get('error', '')}")

    if report["weak"]:
        st.divider()
        st.subheader("📖 Study Plan")
        for t in report["weak"]:
            with st.expander(f"📚 Improve: {t}"):
                st.markdown(
                    f"1. Revise core **{t}** concepts\n"
                    f"2. Build a small project using **{t}**\n"
                    f"3. Practice more **{t}** questions here\n"
                    f"4. Re-read the 'stronger answers' above"
                )

    st.divider()
    if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
        for _key in list(st.session_state.keys()):
            del st.session_state[_key]
        st.rerun()

st.divider()
st.markdown(
    "<center>🎯 AI Mock Interviewer | LangChain + Groq + Whisper + Streamlit | "
    "Navyasri Akula</center>",
    unsafe_allow_html=True,
)