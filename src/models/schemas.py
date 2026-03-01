from pydantic import BaseModel, Field


class TestCase(BaseModel):
    query: str
    ground_truth: str


class EvalResult(BaseModel):
    query: str
    answer: str
    chunks: list[str]
    ground_truth: str
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    context_recall: float = Field(default=0.0, ge=0.0, le=1.0)


class EvalReport(BaseModel):
    results: list[EvalResult]
    avg_faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_context_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    weak_spots: list[str] = Field(default_factory=list)
