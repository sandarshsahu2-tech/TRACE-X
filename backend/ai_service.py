from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# EXPLICIT ENVIRONMENT LOADING
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


# ============================================================
# TRACE-X GENAI SERVICE
# ============================================================

class AIInvestigationService:

    def __init__(self) -> None:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from "
                f"{ENV_FILE}"
            )

        # IMPORTANT:
        # Use the current stable Gemini model directly.
        # Do not fall back to an old model.
        self.model = "gemini-3.6-flash"

        self.client = genai.Client(
            api_key=api_key
        )

        print(
            "TRACE-X Gemini model:",
            self.model,
        )

    # ========================================================
    # SYSTEM INSTRUCTION
    # ========================================================

    @staticmethod
    def _system_instruction() -> str:

        return """
You are the TRACE-X Investigation Copilot.

TRACE-X V1 is the authoritative financial-risk
decision engine.

Your job is to explain evidence already produced
by TRACE-X to a human investigator.

STRICT RULES:

1. Use ONLY supplied evidence.
2. Never invent transactions.
3. Never invent accounts.
4. Never invent relationships.
5. Never invent rules.
6. Never invent numerical values.
7. Never change the TRACE-X decision.
8. Never generate a new TRACE-X risk score.
9. Do not claim certainty beyond the evidence.
10. Distinguish observed evidence from interpretation.

Return ONLY JSON in this structure:

{
  "summary": "string",
  "risk_level": "HIGH | MEDIUM | NORMAL | UNKNOWN",
  "why_flagged": ["string"],
  "strongest_evidence": [
    {
      "signal": "string",
      "value": "string",
      "source": "string"
    }
  ],
  "recommended_actions": ["string"],
  "follow_up_questions": ["string"],
  "disclaimer": "string"
}

If evidence is insufficient, explicitly say so.
"""

    # ========================================================
    # INVESTIGATION
    # ========================================================

    def investigate(
        self,
        evidence_pack: dict[str, Any],
    ) -> dict[str, Any]:

        prompt = (
            "Analyze this TRACE-X evidence pack.\n\n"
            "Use ONLY supplied evidence.\n"
            "Do not invent missing information.\n"
            "Do not change the TRACE-X decision.\n\n"
            "EVIDENCE PACK:\n"
            + json.dumps(
                evidence_pack,
                indent=2,
                default=str,
            )
        )

        response = (
            self.client
            .models
            .generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        self._system_instruction()
                    ),
                    response_mime_type=(
                        "application/json"
                    ),
                ),
            )
        )

        if response is None:
            raise RuntimeError(
                "Gemini returned no response."
            )

        output = getattr(
            response,
            "text",
            None,
        )

        if not output:
            raise RuntimeError(
                "Gemini returned empty output."
            )

        try:
            result = json.loads(
                output
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemini returned invalid JSON."
            ) from exc

        self._validate_result(
            result
        )

        return result

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_result(
        result: dict[str, Any],
    ) -> None:

        required = [
            "summary",
            "risk_level",
            "why_flagged",
            "strongest_evidence",
            "recommended_actions",
            "follow_up_questions",
            "disclaimer",
        ]

        missing = [
            field
            for field in required
            if field not in result
        ]

        if missing:
            raise ValueError(
                "Gemini response missing: "
                + ", ".join(missing)
            )

        for field in [
            "why_flagged",
            "strongest_evidence",
            "recommended_actions",
            "follow_up_questions",
        ]:

            if not isinstance(
                result[field],
                list,
            ):
                raise ValueError(
                    f"{field} must be a list."
                )


# ============================================================
# LAZY SINGLETON
# ============================================================

_ai_service: AIInvestigationService | None = None


def get_ai_investigation_service() -> AIInvestigationService:

    global _ai_service

    if _ai_service is None:
        _ai_service = (
            AIInvestigationService()
        )

    return _ai_service