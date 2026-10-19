"""ROUGE and limited text diagnostics; none of these certifies clinical safety."""

from collections import Counter
import re

CONCEPTS = {
    "pneumothorax": r"pneumothorax|collapsed lung",
    "effusion": r"(?:pleural )?effusion|fluid (?:around|outside) (?:the )?lungs?",
    "pneumonia": r"pneumonia|lung infection",
    "nodule": r"nodule|small (?:lung )?(?:lump|spot)",
    "fracture": r"fracture|broken (?:rib|bone)",
    "cardiomegaly": r"cardiomegaly|enlarged heart",
    "oedema": r"edema|oedema|fluid in (?:the )?lungs?",
}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]


def concept_states(text):
    """A deliberately small cue matcher for choosing cases for manual review.

    Negation scope, synonym coverage and coreference are incomplete. These flags
    are review prompts, not a clinical error rate or reference-free truth metric.
    """
    states = {}
    for sentence in sentences(text.casefold()):
        for concept, expression in CONCEPTS.items():
            for match in re.finditer(r"\b(?:" + expression + r")\b", sentence):
                prefix = sentence[max(0, match.start() - 70):match.start()]
                prefix = re.split(r"\b(?:but|however|although)\b|[;:]", prefix)[-1]
                if re.search(r"\b(no|without|absent|negative for|no evidence of)\b", prefix):
                    state = "negated"
                elif re.search(r"\b(possible|possibly|suspected|suggestive|may|might|cannot exclude|rule out)\b", prefix):
                    state = "uncertain"
                else:
                    state = "affirmed"
                states.setdefault(concept, set()).add(state)
    return {key: sorted(value) for key, value in states.items()}


def safety_flags(source, prediction):
    before, after = concept_states(source), concept_states(prediction)
    return {"missing_concepts": sorted(set(before) - set(after)),
            "added_concepts": sorted(set(after) - set(before)),
            "changed_states": {c: {"input": before[c], "prediction": after[c]}
                               for c in sorted(set(before) & set(after)) if before[c] != after[c]}}


def score_predictions(rows):
    from rouge_score import rouge_scorer
    if not rows:
        raise ValueError("Cannot score an empty prediction file")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True,
                                    split_summaries=False)
    totals = Counter()
    for row in rows:
        # Explicit sentence boundaries for union-LCS ROUGE-Lsum. This avoids
        # needing an unversioned NLTK punkt download in a training runtime.
        reference = "\n".join(sentences(row["target"]))
        prediction = "\n".join(sentences(row["prediction"]))
        for name, score in scorer.score(reference, prediction).items():
            totals[name] += score.fmeasure
    return {key: value / len(rows) for key, value in sorted(totals.items())}
