
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


class EvidenceChecker:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.eval()

        self.id2label = {
            int(key): str(value).lower()
            for key, value in self.model.config.id2label.items()
        }

    @staticmethod
    def split_claims(answer):
        """Basic sentence splitting; complex claims may need manual review."""

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            answer.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def check_claim(self, evidence, claim):
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(
                logits, dim=-1
            ).tolist()

        scores = {
            self.id2label.get(i, f"label_{i}"): float(score)
            for i, score in enumerate(probabilities)
        }

        def find_score(word):
            for label, score in scores.items():
                if word in label:
                    return score
            return None

        entailment = find_score("entail")
        contradiction = find_score("contrad")
        neutral = find_score("neutral")

        # Avoid guessing if model labels are unexpected.
        if any(
            score is None
            for score in (
                entailment,
                contradiction,
                neutral,
            )
        ):
            return {
                "claim": claim,
                "status": "Needs manual review",
                "confidence": max(scores.values()),
            }

        options = [
            ("Supported by retrieved evidence", entailment),
            ("Contradicted by retrieved evidence", contradiction),
            ("Insufficient evidence", neutral),
        ]

        status, score = max(
            options,
            key=lambda item: item[1],
        )

        return {
            "claim": claim,
            "status": status,
            "confidence": score,
        }

    def check_answer(self, evidence, answer):
        """Check every sentence against the supplied evidence."""

        claims = self.split_claims(answer)

        return [
            self.check_claim(evidence, claim)
            for claim in claims
        ]
