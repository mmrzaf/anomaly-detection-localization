# anomaly-detection-localization

FastAPI service for anomaly detection + localization inference.

## Features
- Health and metrics endpoints
- Model registry endpoint
- Image prediction endpoint (supports `padim` and `student`)
- Minimal demo UI at `/`

## Run locally
```bash
poetry install
poetry run uvicorn app.main:app --reload
````

Open:

* UI: `http://localhost:8000/`
* Docs: `http://localhost:8000/docs`

## API endpoints

* `GET /api/health/`
* `GET /api/metrics/`
* `GET /api/v1/models`
* `POST /api/v1/predict`

## Predict example (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "image=@sample.jpg" \
  -F "method=padim" \
  -F "category=bottle" \
  -F "return_visuals=false"
```

## Notes

* Max upload size is controlled by `request_max_bytes` in settings.
* `return_visuals=false` reduces response payload size.# anomaly-detection-localization
