FROM python:3.11-slim

# libgl/libglib: matplotlib + PIL image handling on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~1 GB instead of ~6 GB. Swap the index URL for
# a CUDA build if the host has a GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.0 torchvision==0.20.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app/dp_cxr_service/

ENV DP_CXR_BUNDLE=/app/dp_cxr_service/bundle \
    DP_CXR_CACHE=/app/.cache \
    PYTHONUNBUFFERED=1

# Bake the frozen encoders into the image so the first request is not a
# half-gigabyte download.
RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')" || true
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); \
    AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')" || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "dp_cxr_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
