from __future__ import annotations

from typing import Any, Dict

ACTION_SET = {
    "accept_and_advance",
    "arithmetic_repair",
    "handle_question",
    "refocus_subgoal",
    "conceptual_hint",
    "ask_for_justification",
}

JUDGE_LABELS = [
    "helpful_progress",
    "partial_progress",
    "no_progress_or_harm",
]

JUDGE_SCORE_MAP = {
    "helpful_progress": 1.0,
    "partial_progress": 0.5,
    "no_progress_or_harm": 0.0,
}

STUDENT_SIM_SYSTEM = """You are simulating a struggling math student in a tutoring dialogue.

Your job is to produce the student's NEXT reply only.

Important behavior rules:
- Do not become suddenly perfect.
- Do not solve the whole problem unless the teacher's move clearly justifies that jump.
- If the teacher's move is vague, unnecessary, or weak, remain confused or only improve slightly.
- Preserve the student's existing misunderstanding unless the teacher directly addresses it.
- Keep the student realistic: hesitant, partial, or stubborn when appropriate.
- Keep the reply short: 1 to 3 sentences maximum.
- Output only the student's next reply, with no extra commentary.
""".strip()

JUDGE_SYSTEM = """You are judging whether a teacher intervention helped the student make progress on the CURRENT tutoring step.

You must choose exactly one label:
- helpful_progress
- partial_progress
- no_progress_or_harm

Definitions:
- helpful_progress:
  The student's next reply shows clear improvement on the current step.
  Examples: becomes correct, more focused, uses the right operation, or meaningfully clarifies the right subgoal.

- partial_progress:
  The student's next reply shows some improvement, but remains incomplete, uncertain, partially correct, or still needs help.

- no_progress_or_harm:
  The student's next reply shows no meaningful improvement, remains off-track, becomes more confused, or the intervention appears unnecessary relative to the current state.

Be strict. Do not reward generic continuation unless it meaningfully improves the student's state on the current step.

Output only one label and nothing else.
""".strip()


def teacher_template(action: str, row: Dict[str, Any]) -> str:
    teacher_question = row.get("teacher_question", "").strip()
    current_subgoal = row.get("current_subgoal", "").strip()

    if action == "accept_and_advance":
        return "Okay, go ahead and continue from here."

    if action == "arithmetic_repair":
        return (
            "Let's slow down and check the arithmetic carefully. "
            "Can you recompute that number step by step?"
        )

    if action == "handle_question":
        return (
            "That's a good question. Let me answer it briefly, then come back to the current step. "
            f"{teacher_question}"
        )

    if action == "refocus_subgoal":
        if teacher_question:
            return f"Let's focus only on the current step: {teacher_question}"
        if current_subgoal:
            return f"Let's focus only on this subgoal: {current_subgoal}"
        return "Let's focus only on the current step."

    if action == "conceptual_hint":
        return (
            "Think about what the problem is asking conceptually. "
            "What relationship or operation makes sense here before calculating?"
        )

    if action == "ask_for_justification":
        return "Can you explain how you got that?"

    raise ValueError(f"Unsupported action: {action}")


def build_student_sim_user(row: Dict[str, Any], teacher_utterance: str) -> str:
    question = row.get("question", "")
    analysis = row.get("analysis", "")
    answer = row.get("answer", "")
    teacher_question = row.get("teacher_question", "")
    student_response = row.get("student_response", "")
    history = row.get("history", [])
    current_subgoal = row.get("current_subgoal", "")

    history_text = []
    for turn in history:
        if isinstance(turn, list) and len(turn) == 2:
            history_text.append(f"Teacher: {turn[0]}\nStudent: {turn[1]}")
    history_block = "\n".join(history_text) if history_text else "None"

    return f"""[PROBLEM]
{question}

[REFERENCE_ANALYSIS]
{analysis}

[REFERENCE_ANSWER]
{answer}

[CURRENT_SUBGOAL]
{current_subgoal}

[HISTORY]
{history_block}

[CURRENT_TEACHER_QUESTION]
{teacher_question}

[CURRENT_STUDENT_RESPONSE]
{student_response}

[TEACHER_NEXT_UTTERANCE]
{teacher_utterance}

Generate ONLY the student's next reply.
"""


def build_judge_user(
    row: Dict[str, Any],
    action: str,
    teacher_utterance: str,
    simulated_student_reply: str,
) -> str:
    question = row.get("question", "")
    teacher_question = row.get("teacher_question", "")
    current_subgoal = row.get("current_subgoal", "")
    original_student_response = row.get("student_response", "")

    return f"""[PROBLEM]
{question}

[CURRENT_SUBGOAL]
{current_subgoal}

[CURRENT_TEACHER_QUESTION]
{teacher_question}

[ORIGINAL_STUDENT_RESPONSE]
{original_student_response}

[TESTED_ACTION]
{action}

[TEACHER_NEXT_UTTERANCE]
{teacher_utterance}

[SIMULATED_NEXT_STUDENT_REPLY]
{simulated_student_reply}

Choose exactly one label:
helpful_progress
partial_progress
no_progress_or_harm
"""