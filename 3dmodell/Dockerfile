FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG HUNYUAN_REF=82920d643c0dc2f7bfd7255f45f62d386edfe60c

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git libgl1 libegl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Hunyuan3D-2.1
RUN git init . \
    && git remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git \
    && git fetch --depth 1 origin "${HUNYUAN_REF}" \
    && git checkout --detach FETCH_HEAD \
    && rm -rf .git \
    && grep -E '^(ninja|pybind11|transformers|diffusers|accelerate|huggingface-hub|safetensors|numpy|scipy|einops|pandas|opencv-python|imageio|scikit-image|rembg|trimesh|pymeshlab|pygltflib|xatlas|omegaconf|pyyaml|configargparse|gradio|fastapi|uvicorn|tqdm|psutil|onnxruntime|pydantic|timm|torchdiffeq)(==|$)' requirements.txt > /tmp/shape-requirements.txt \
    && pip install --no-cache-dir -r /tmp/shape-requirements.txt \
    && rm /tmp/shape-requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    HY3DGEN_MODELS=/data/models \
    U2NET_HOME=/data/rembg \
    XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/torch \
    GRADIO_TEMP_DIR=/data/gradio \
    TMPDIR=/data/tmp

EXPOSE 7860

CMD ["/bin/bash", "-c", "mkdir -p \"$HF_HOME\" \"$HY3DGEN_MODELS\" \"$U2NET_HOME\" \"$XDG_CACHE_HOME\" \"$TORCH_HOME\" \"$GRADIO_TEMP_DIR\" \"$TMPDIR\" /data/outputs && exec python3 gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 --disable_tex --cache-path /data/outputs --host 0.0.0.0 --port 7860"]
