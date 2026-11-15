from app.explainability.exceptions import ExplainabilityError
from app.schemas.explainability import ExplainabilityRequest, ExplainabilityResponse
from app.services.explainability_service import ExplainabilityService
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/explainability", tags=["Explainability"])


@router.post("/", response_model=ExplainabilityResponse)  # type: ignore[misc]
async def explain(request: ExplainabilityRequest) -> ExplainabilityResponse:
    try:
        service = ExplainabilityService()
        response = service.explain(request)
        return response
    except ExplainabilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
