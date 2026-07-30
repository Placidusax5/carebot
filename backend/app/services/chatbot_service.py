"""
AI Healthcare Chatbot Module (Module 2) - rule-based NLP engine.

This runs entirely offline with no external LLM dependency: it extracts
symptoms from free-text messages via keyword/synonym matching, asks
clarifying follow-up questions when it doesn't have enough information,
and calls the ML disease-prediction service once it has enough signal.

Swap-in point: if OPENAI_API_KEY / ANTHROPIC_API_KEY is configured in
.env, a real LLM client could replace `extract_symptoms_from_text` and
`generate_reply` below without changing the router or response schema.
"""
import re
from typing import List, Tuple

from app.services import ml_service

# Hand-authored synonyms for the most commonly-phrased symptoms, layered on
# top of automatic underscore->space matching against the full symptom list.
SYNONYMS = {
    "fever": ["fever", "high temperature", "temperature", "running a fever", "hot"],
    "high_fever": ["high fever", "very high temperature", "burning up"],
    "mild_fever": ["mild fever", "slight fever", "low grade fever"],
    "headache": ["headache", "head pain", "head hurts", "my head hurts"],
    "severe_headache": ["severe headache", "really bad headache", "splitting headache", "worst headache"],
    "mild_headache": ["mild headache", "slight headache", "dull headache"],
    "sore_throat": ["sore throat", "throat hurts", "throat pain", "scratchy throat"],
    "runny_nose": ["runny nose", "nose is running", "nasal discharge"],
    "sneezing": ["sneezing", "sneeze a lot", "keep sneezing"],
    "mild_cough": ["mild cough", "slight cough"],
    "dry_cough": ["dry cough", "cough with no mucus"],
    "productive_cough": ["cough with mucus", "coughing up phlegm", "wet cough", "productive cough"],
    "persistent_cough": ["persistent cough", "cough that won't go away", "constant cough"],
    "cough": ["cough", "coughing"],
    "vomiting": ["vomiting", "throwing up", "puking", "vomit"],
    "nausea": ["nausea", "nauseous", "feel sick to my stomach", "queasy"],
    "diarrhea": ["diarrhea", "loose stools", "watery stool"],
    "constipation": ["constipation", "can't poop", "difficulty passing stool"],
    "abdominal_pain": ["stomach ache", "abdominal pain", "belly pain", "tummy hurts", "stomach pain", "stomach hurts", "belly hurts", "stomach ache", "tummy ache"],
    "abdominal_cramps": ["stomach cramps", "abdominal cramps", "cramping"],
    "chest_pain": ["chest pain", "chest hurts", "pain in my chest"],
    "chest_tightness": ["chest tightness", "tight chest", "chest feels tight"],
    "shortness_of_breath": ["shortness of breath", "can't breathe", "trouble breathing", "breathless", "out of breath"],
    "difficulty_breathing": ["difficulty breathing", "hard to breathe", "struggling to breathe"],
    "wheezing": ["wheezing", "whistling sound breathing"],
    "fatigue": ["fatigue", "tired", "exhausted", "no energy", "low energy"],
    "body_ache": ["body ache", "aches all over", "everything hurts", "muscle aches"],
    "muscle_pain": ["muscle pain", "muscles hurt", "sore muscles"],
    "joint_pain": ["joint pain", "joints hurt", "achy joints"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "feel faint"],
    "chills": ["chills", "shivering", "feeling cold"],
    "sweating": ["sweating a lot", "excessive sweating", "night sweats"],
    "rash": ["rash", "skin rash", "red spots", "breaking out"],
    "itchy_skin": ["itchy skin", "skin itching", "itching"],
    "loss_of_taste": ["loss of taste", "can't taste", "no sense of taste"],
    "loss_of_smell": ["loss of smell", "can't smell", "no sense of smell"],
    "frequent_urination": ["peeing a lot", "frequent urination", "urinating often"],
    "burning_urination": ["burning when i pee", "painful urination", "burning urination"],
    "weight_loss": ["losing weight", "weight loss", "lost weight"],
    "weight_gain": ["gaining weight", "weight gain", "gained weight"],
    "blurred_vision": ["blurry vision", "blurred vision", "can't see clearly"],
    "persistent_sadness": ["feeling sad", "depressed", "down all the time", "persistent sadness"],
    "excessive_worry": ["worried all the time", "anxious", "excessive worry", "can't stop worrying"],
    "difficulty_falling_asleep": ["can't fall asleep", "trouble sleeping", "difficulty falling asleep"],
}

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
THANKS = ["thanks", "thank you", "appreciate it", "cheers"]
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "difficulty breathing", "severe bleeding",
    "suicidal", "unconscious", "stroke", "heart attack", "not breathing",
]

MIN_SYMPTOMS_FOR_PREDICTION = 3

# Words that flip a nearby symptom to "not present" ("I don't have a fever").
# Apostrophes are stripped by _normalize, so contractions appear split
# (e.g. "don't" -> "don t"); we match the leading fragment ("don").
NEGATION_CUES = {
    "no", "not", "never", "without", "nor", "denies", "deny", "negative",
    "don", "doesn", "didn", "dont", "havent", "haven", "hasnt", "hasn",
    "isn", "arent", "aren", "wasn", "weren",
}
# Tokens that end a negation's reach, so "no cough but a headache" keeps the
# headache. Conjunctions reset the scope back to "present unless negated".
NEGATION_SCOPE_BREAKERS = {
    "but", "however", "except", "though", "although", "yet",
    "and", "or", "plus", "also", "still", "with", "just", "only",
}
# How many words before a symptom we scan for a negation cue.
NEGATION_WINDOW = 4

# Punctuation that separates independent clauses, so "no fever, sore throat"
# doesn't let the negation bleed onto "sore throat".
_CLAUSE_SPLIT_RE = re.compile(r"[,.;:!?]")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _clauses(text: str) -> List[str]:
    """Break text into clause-sized, normalized chunks on punctuation so a
    negation in one clause can't reach a symptom in another."""
    return [_normalize(chunk) for chunk in _CLAUSE_SPLIT_RE.split(text.lower())]


def _is_negated(normalized: str, phrase: str) -> bool:
    """True only if *every* occurrence of `phrase` sits inside a negation
    scope. A single un-negated mention is enough to count the symptom as
    present (people often repeat or clarify)."""
    start = 0
    occurrences = 0
    negated = 0
    while True:
        i = normalized.find(phrase, start)
        if i == -1:
            break
        occurrences += 1
        preceding = normalized[:i].split()
        for tok in reversed(preceding[-NEGATION_WINDOW:]):
            if tok in NEGATION_SCOPE_BREAKERS:
                break
            if tok in NEGATION_CUES:
                negated += 1
                break
        start = i + len(phrase)
    return occurrences > 0 and negated == occurrences


def extract_symptoms_from_text(text: str) -> List[str]:
    """Keyword/synonym-based symptom extraction (no external LLM needed).

    Negation-aware: phrases like "no fever" or "I don't have a cough" are
    skipped so we don't record symptoms the user explicitly denies."""
    found = set()
    known_symptoms = ml_service.get_known_symptoms()

    for clause in _clauses(text):
        if not clause:
            continue

        # 1. Hand-authored synonym phrases (checked first, most natural-sounding)
        for symptom_id, phrases in SYNONYMS.items():
            for phrase in phrases:
                if phrase in clause and not _is_negated(clause, phrase):
                    found.add(symptom_id)
                    break

        # 2. Fallback: literal "symptom_id with underscores replaced by spaces"
        for symptom_id in known_symptoms:
            phrase = symptom_id.replace("_", " ")
            if phrase in clause and not _is_negated(clause, phrase):
                found.add(symptom_id)

    return sorted(found)


def is_emergency(text: str) -> bool:
    normalized = _normalize(text)
    return any(kw in normalized for kw in EMERGENCY_KEYWORDS)


def _risk_guidance(risk_level: str) -> str:
    """Plain-language next-step advice matched to the assessed risk level."""
    return {
        "High": (
            "Because this carries a higher risk, I'd strongly recommend seeing a "
            "doctor soon — and seek urgent care right away if symptoms are severe."
        ),
        "Medium": (
            "It's a good idea to keep an eye on your symptoms and see a doctor if "
            "they persist for more than a few days or start getting worse."
        ),
        "Low": (
            "This usually isn't serious and often improves with rest, but do check "
            "with a doctor if it lingers beyond a week or worsens."
        ),
    }.get(risk_level, "Please consult a doctor if you're concerned or symptoms worsen.")


def _suggest_followup(symptoms_so_far: List[str]) -> str:
    prompts = [
        "Can you tell me more about how you're feeling? For example, do you have a fever, cough, or pain anywhere?",
        "Got it. Any other symptoms — nausea, fatigue, headache, or anything else out of the ordinary?",
        "Thanks for sharing that. Is there anything else you've noticed, like changes in appetite, sleep, or energy levels?",
        "Okay. On a scale of mild to severe, how would you describe your main symptom, and when did it start?",
    ]
    idx = min(len(symptoms_so_far), len(prompts) - 1)
    return prompts[idx]


def handle_message(message: str, prior_symptoms: List[str]) -> Tuple[str, List[str], str, dict]:
    """
    Returns: (reply_text, all_extracted_symptoms, followup_question_or_empty, prediction_dict_or_empty)
    """
    normalized = _normalize(message)

    if is_emergency(message):
        reply = (
            "Some of what you're describing can be a medical emergency (e.g. chest pain or "
            "difficulty breathing). Please call your local emergency number or go to the "
            "nearest emergency room right away. I can still help with general guidance, but "
            "please prioritize getting immediate in-person care."
        )
        return reply, prior_symptoms, "", {}

    if any(g in normalized for g in GREETINGS) and len(normalized.split()) <= 4:
        return (
            "Hi, I'm CareBot. I can help you understand possible causes for your symptoms. "
            "What symptoms are you experiencing today?",
            prior_symptoms, "", {}
        )

    if any(t in normalized for t in THANKS):
        return ("You're welcome! Take care, and don't hesitate to check in again if anything changes.",
                prior_symptoms, "", {})

    new_symptoms = extract_symptoms_from_text(message)
    all_symptoms = sorted(set(prior_symptoms) | set(new_symptoms))

    if not new_symptoms and not all_symptoms:
        return (
            "I didn't quite catch specific symptoms in that. Could you describe what you're "
            "feeling? For example: \"I have a fever and sore throat\" or \"my stomach hurts and I feel nauseous.\"",
            all_symptoms, "", {}
        )

    if len(all_symptoms) < MIN_SYMPTOMS_FOR_PREDICTION:
        followup = _suggest_followup(all_symptoms)
        symptom_list_str = ", ".join(s.replace("_", " ") for s in all_symptoms) or "none yet"
        reply = f"Noted — so far I have: {symptom_list_str}. {followup}"
        return reply, all_symptoms, followup, {}

    prediction = ml_service.predict_disease(all_symptoms)
    top = prediction["top_prediction"]
    conf = prediction["predictions"][0]["confidence"] if prediction["predictions"] else 0
    symptom_list_str = ", ".join(s.replace("_", " ") for s in all_symptoms)

    remedy_text = ""
    remedies = prediction.get("home_remedies", [])
    if remedies:
        remedy_lines = "\n".join(f"• {r}" for r in remedies[:3])
        remedy_text = f"\n\nA few things that may help you feel better at home:\n{remedy_lines}"

    reply = (
        f"Based on what you've described ({symptom_list_str}), the most likely "
        f"condition is **{top}** ({conf}% confidence), with a **{prediction['risk_level']}** "
        f"risk level. {prediction['description']}"
        f"{remedy_text}\n\n{_risk_guidance(prediction['risk_level'])} "
        f"This isn't a diagnosis — please check with a doctor to confirm, and seek care "
        f"sooner if your symptoms get worse."
    )
    return reply, all_symptoms, "", prediction
