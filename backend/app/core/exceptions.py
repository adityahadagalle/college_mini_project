from app.core.logging_config import logger
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)  # type: ignore[misc]
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "N/A")
        logger.warning(f"Validation Error [correlation_id={correlation_id}]: " f"{exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "error_type": "VALIDATION_ERROR",
                "detail": exc.errors(),
                "correlation_id": correlation_id,
            },
        )

    @app.exception_handler(Exception)  # type: ignore[misc]
    async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "N/A")
        logger.error(
            f"Unexpected Exception [correlation_id={correlation_id}]: " f"{str(exc)}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_type": "INTERNAL_SERVER_ERROR",
                "detail": "An unexpected internal server error occurred.",
                "correlation_id": correlation_id,
            },
        )
