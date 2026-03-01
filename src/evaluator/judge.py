import json
import logging

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


FAITHFULNESS_SYSTEM_PROMPT = """You are an evaluation judge. Your task is to assess the faithfulness of an answer.
Faithfulness measures whether EVERY claim in the answer is supported by the provided context chunks.
A score of 1.0 means all claims are fully supported. A score of 0.0 means none are supported.
Evaluate carefully, then return ONLY a JSON object with a single key "score" containing a float between 0.0 and 1.0.
No explanation, no other text. Example: {"score": 0.85}"""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an evaluation judge. Your task is to assess the relevance of an answer to a question.
Answer relevance measures how well the answer addresses the specific question asked.
A score of 1.0 means the answer perfectly addresses the question. A score of 0.0 means the answer is completely off-topic.
Evaluate carefully, then return ONLY a JSON object with a single key "score" containing a float between 0.0 and 1.0.
No explanation, no other text. Example: {"score": 0.85}"""

CONTEXT_PRECISION_SYSTEM_PROMPT = """You are an evaluation judge. Your task is to assess context precision.
Context precision measures how many of the retrieved chunks are actually relevant to answering the query.
A score of 1.0 means all chunks are relevant. A score of 0.0 means none are relevant.
Evaluate carefully, then return ONLY a JSON object with a single key "score" containing a float between 0.0 and 1.0.
No explanation, no other text. Example: {"score": 0.85}"""

CONTEXT_RECALL_SYSTEM_PROMPT = """You are an evaluation judge. Your task is to assess context recall.
Context recall measures whether the retrieved chunks contain all the information needed to produce the ground truth answer.
A score of 1.0 means the chunks contain all necessary information. A score of 0.0 means they contain none.
Evaluate carefully, then return ONLY a JSON object with a single key "score" containing a float between 0.0 and 1.0.
No explanation, no other text. Example: {"score": 0.85}"""


class OllamaJudge:
    """LLM-as-Judge using ChatOllama for structured scoring."""

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)

    def score(self, system_prompt: str, user_prompt: str) -> float:
        """Send a prompt to the judge and return a float score 0.0-1.0.

        Handles JSON parsing failures gracefully with a default score of 0.0.
        """
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Try to extract JSON from potentially wrapped response
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]

            parsed = json.loads(content)
            raw_score = float(parsed["score"])
            # Clamp to [0.0, 1.0]
            return max(0.0, min(1.0, raw_score))

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse judge response: {e}. Defaulting to 0.0")
            return 0.0
        except Exception as e:
            logger.warning(f"Judge invocation failed: {e}. Defaulting to 0.0")
            return 0.0
