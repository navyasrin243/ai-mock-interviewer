from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from question_gen import (generate_questions,
                           generate_followup_question,
                           extract_topics_from_resume)
from evaluator import evaluate_answer

load_dotenv()


class InterviewState(TypedDict):
    resume_text:          str
    jd_text:              str
    interview_type:       str
    num_questions:        int
    topics_to_cover:      List[str]
    current_topic_idx:    int
    questions_on_topic:   int
    is_followup:          bool
    question_set:         Optional[dict]
    current_index:        int
    current_question:     Optional[dict]
    answers:              List[str]
    evaluations:          List[dict]
    weak_topics:          List[str]
    strong_topics:        List[str]
    next_difficulty:      str
    interview_done:       bool
    asked_count:          int
    evaluated_count:      int
    failed_count:         int
    error:                Optional[str]


def validate_inputs(resume_text, jd_text, answers,
                    interview_type, num_questions) -> List[str]:
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text required.")
    if not isinstance(jd_text, str) or not jd_text.strip():
        raise ValueError("JD required.")
    if isinstance(num_questions, bool) or \
       not isinstance(num_questions, int) or \
       not 1 <= num_questions <= 10:
        raise ValueError("num_questions must be integer 1-10.")
    if not isinstance(answers, list):
        raise ValueError("answers must be list.")
    if not all(isinstance(a, str) for a in answers):
        raise ValueError("All answers must be strings.")
    normalized = answers[:num_questions]
    if len(normalized) < num_questions:
        normalized.extend([""] * (num_questions - len(normalized)))
    return normalized


def generate_questions_node(state: InterviewState) -> dict:
    print("\n🎯 NODE: generate_questions")
    try:
        result = generate_questions(
            resume_text=state["resume_text"],
            jd_text=state["jd_text"],
            interview_type=state["interview_type"],
            num_questions=state["num_questions"]
        )
        questions = [
            {"question": q.question, "topic": q.topic,
             "difficulty": q.difficulty, "q_type": q.q_type,
             "ideal_points": q.ideal_points,
             "is_followup": False}
            for q in result.questions
        ]
        from langchain_groq import ChatGroq
        import os
        llm = ChatGroq(model="qwen/qwen3.8-27b",
                       groq_api_key=os.getenv("GROQ_API_KEY"),
                       temperature=0.1)
        topics = extract_topics_from_resume(state["resume_text"], llm)
        print(f"📋 Topics: {topics}")
        return {
            "question_set": {
                "candidate_name": result.candidate_name,
                "target_role":    result.target_role,
                "key_topics":     result.key_topics,
                "questions":      questions
            },
            "topics_to_cover":    topics,
            "current_topic_idx":  0,
            "questions_on_topic": 0,
            "is_followup":        False,
            "current_index":      0,
            "current_question":   None,
            "error":              None
        }
    except Exception as e:
        msg = f"Question generation failed: {e}"
        print(f"❌ {msg}")
        return {"question_set": None, "error": msg, "interview_done": True}


def ask_question_node(state: InterviewState) -> dict:
    idx       = state["current_index"]
    questions = state["question_set"]["questions"]
    print(f"\n🎯 NODE: ask_question (index={idx})")
    if idx >= len(questions):
        return {"interview_done": True, "current_question": None}
    q = dict(questions[idx])
    print(f"📌 Q{idx+1} [{q['q_type'].upper()}|{q['difficulty']}]: "
          f"{q['question'][:80]}")
    return {
        "current_question": q,
        "asked_count":      state.get("asked_count", 0) + 1,
        "interview_done":   False
    }


def evaluate_answer_node(state: InterviewState) -> dict:
    print("\n🎯 NODE: evaluate_answer")
    q        = state["current_question"]
    idx      = state["current_index"]
    user_ans = state["answers"][idx] if idx < len(state["answers"]) else ""
    try:
        result = evaluate_answer(
            question=q["question"],
            user_answer=user_ans,
            ideal_points=q["ideal_points"],
            q_type=q["q_type"],
            resume_context=state["resume_text"][:300]
        )
        weak   = list(state.get("weak_topics", []))
        strong = list(state.get("strong_topics", []))
        if result.final_score <= 4 and q["topic"] not in weak:
            weak.append(q["topic"])
        elif result.final_score >= 8 and q["topic"] not in strong:
            strong.append(q["topic"])
        evals = list(state.get("evaluations", []))
        evals.append({
            "question":        q["question"],
            "topic":           q["topic"],
            "q_type":          q["q_type"],
            "difficulty":      q["difficulty"],
            "is_followup":     q.get("is_followup", False),
            "user_answer":     user_ans,
            "correctness":     result.correctness,
            "completeness":    result.completeness,
            "relevance":       result.relevance,
            "final_score":     result.final_score,
            "good_points":     result.good_points,
            "missing_points":  result.missing_points,
            "better_answer":   result.better_answer,
            "next_difficulty": result.next_difficulty,
            "status":          "evaluated"
        })
        return {
            "evaluations":     evals,
            "weak_topics":     weak,
            "strong_topics":   strong,
            "next_difficulty": result.next_difficulty,
            "evaluated_count": state.get("evaluated_count", 0) + 1,
            "error":           None
        }
    except Exception as e:
        print(f"⚠️ Eval error: {e}")
        evals = list(state.get("evaluations", []))
        evals.append({
            "question": q["question"], "topic": q["topic"],
            "user_answer": user_ans, "final_score": None,
            "status": "failed", "error": str(e)
        })
        return {"evaluations":  evals,
                "failed_count": state.get("failed_count", 0) + 1}


def decide_next_node(state: InterviewState) -> dict:
    print("\n🎯 NODE: decide_next")
    evaluations     = state.get("evaluations", [])
    questions       = state["question_set"]["questions"]
    current_idx     = state["current_index"]
    topics          = state.get("topics_to_cover", [])
    topic_idx       = state.get("current_topic_idx", 0)
    is_followup     = state.get("is_followup", False)
    last_eval       = evaluations[-1] if evaluations else {}
    last_score      = last_eval.get("final_score")
    if last_score is None: last_score = 5
    current_topic   = topics[topic_idx] if topic_idx < len(topics) else None

    print(f"   Topic: {current_topic} | Score: {last_score} | Followup: {is_followup}")

    # Already did follow-up → next topic
    if is_followup:
        return _move_to_next_topic(state, current_idx, topic_idx, topics, questions)

    # Score < 5 → generate follow-up
    if isinstance(last_score, int) and last_score < 5:
        return _generate_followup(state, current_idx, last_eval, current_topic)

    # Score >= 5 → next topic
    return _move_to_next_topic(state, current_idx, topic_idx, topics, questions)


def _generate_followup(state, current_idx, last_eval, current_topic):
    try:
        from langchain_groq import ChatGroq
        import os
        llm = ChatGroq(model="qwen/qwen3.8-27b",
                       groq_api_key=os.getenv("GROQ_API_KEY"),
                       temperature=0.5)
        fq = generate_followup_question(
            original_question=last_eval.get("question", ""),
            user_answer=last_eval.get("user_answer", ""),
            missing_points=last_eval.get("missing_points", []),
            topic=current_topic or "General",
            resume_text=state["resume_text"],
            llm=llm
        )
        questions = list(state["question_set"]["questions"])
        questions.insert(current_idx + 1, fq)
        return {
            "question_set": {**state["question_set"], "questions": questions},
            "current_index":      current_idx + 1,
            "questions_on_topic": state.get("questions_on_topic", 0) + 1,
            "is_followup":        True,
            "interview_done":     False
        }
    except Exception as e:
        print(f"⚠️ Follow-up failed: {e} → Next topic")
        return _move_to_next_topic(
            state, current_idx,
            state.get("current_topic_idx", 0),
            state.get("topics_to_cover", []),
            state["question_set"]["questions"]
        )


def _move_to_next_topic(state, current_idx, topic_idx, topics, questions):
    next_topic_idx = topic_idx + 1
    if next_topic_idx >= len(topics):
        print("   ✅ All topics covered → Report!")
        return {
            "current_index":      current_idx + 1,
            "current_topic_idx":  next_topic_idx,
            "questions_on_topic": 0,
            "is_followup":        False,
            "interview_done":     True
        }
    next_topic = topics[next_topic_idx]
    next_q_idx = current_idx + 1
    for i in range(current_idx + 1, len(questions)):
        if questions[i]["topic"] == next_topic:
            next_q_idx = i
            break
    print(f"   → Next topic: {next_topic} (Q{next_q_idx+1})")
    return {
        "current_index":      next_q_idx,
        "current_topic_idx":  next_topic_idx,
        "questions_on_topic": 0,
        "is_followup":        False,
        "interview_done":     False
    }


def generate_report_node(state: InterviewState) -> dict:
    print("\n🎯 NODE: generate_report")
    evals      = list(state.get("evaluations", []))
    successful = [e for e in evals
                  if e.get("status") == "evaluated"
                  and isinstance(e.get("final_score"), int)]
    failed     = [e for e in evals if e.get("status") == "failed"]
    scores     = [e["final_score"] for e in successful]
    if scores:
        overall = round(sum(scores) / len(scores), 1)
        if overall >= 8:   level = "Excellent 🌟"
        elif overall >= 6: level = "Good ✅"
        elif overall >= 4: level = "Average ⚠️"
        else:              level = "Needs Improvement ❌"
    else:
        overall, level = None, "Unavailable"
    topic_scores = {}
    for e in successful:
        t = e.get("topic", "Unknown")
        topic_scores.setdefault(t, []).append(e["final_score"])
    topic_summary = {
        t: round(sum(s)/len(s), 1)
        for t, s in topic_scores.items()
    }
    print(f"📊 Score: {overall}/10 — {level}")
    return {
        "question_set": {
            **state["question_set"],
            "report": {
                "overall_score":   overall,
                "level":           level,
                "asked_count":     state.get("asked_count", 0),
                "evaluated_count": len(successful),
                "failed_count":    len(failed),
                "strong_topics":   list(state.get("strong_topics", [])),
                "weak_topics":     list(state.get("weak_topics", [])),
                "topic_scores":    topic_summary,
                "evaluations":     evals,
            }
        },
        "interview_done": True
    }


def after_generation(state): return "end" if state.get("error") else "ask"
def after_decide(state):
    if state.get("error"):          return "end"
    if state.get("interview_done"): return "report"
    return "ask"


def build_interview_graph():
    g = StateGraph(InterviewState)
    g.add_node("generate_questions", generate_questions_node)
    g.add_node("ask_question",       ask_question_node)
    g.add_node("evaluate_answer",    evaluate_answer_node)
    g.add_node("decide_next",        decide_next_node)
    g.add_node("generate_report",    generate_report_node)
    g.set_entry_point("generate_questions")
    g.add_conditional_edges("generate_questions", after_generation,
                             {"ask": "ask_question", "end": END})
    g.add_edge("ask_question",    "evaluate_answer")
    g.add_edge("evaluate_answer", "decide_next")
    g.add_conditional_edges("decide_next", after_decide,
                             {"ask": "ask_question",
                              "report": "generate_report", "end": END})
    g.add_edge("generate_report", END)
    return g.compile()


def run_interview(resume_text, jd_text, answers,
                  interview_type="mixed", num_questions=5) -> dict:
    normalized = validate_inputs(
        resume_text, jd_text, answers, interview_type, num_questions
    )
    initial: InterviewState = {
        "resume_text":        resume_text, "jd_text": jd_text,
        "interview_type":     interview_type, "num_questions": num_questions,
        "topics_to_cover":    [], "current_topic_idx": 0,
        "questions_on_topic": 0, "is_followup": False,
        "question_set":       None, "current_index": 0,
        "current_question":   None, "answers": normalized,
        "evaluations":        [], "weak_topics": [],
        "strong_topics":      [], "next_difficulty": "same",
        "interview_done":     False, "asked_count": 0,
        "evaluated_count":    0, "failed_count": 0, "error": None
    }
    print("\n" + "="*50)
    print("🚀 AI Mock Interviewer — Dynamic Mode")
    print("="*50)
    return build_interview_graph().invoke(initial)