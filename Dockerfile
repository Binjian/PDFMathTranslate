FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1 \
     UV_LINK_MODE=copy \
     HOME=/root

# # Download all required fonts
# ADD "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifCN-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifTW-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifJP-Regular.ttf" /app/
# ADD "https://github.com/timelic/source-han-serif/releases/download/main/SourceHanSerifKR-Regular.ttf" /app/

RUN apt-get update && \
     apt-get install --no-install-recommends -y \
     libgl1 \
     libglib2.0-0 \
     libxext6 \
     libsm6 \
     libxrender1 \
     libreoffice-core \
     libreoffice-writer && \
     rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml && babeldoc --version && babeldoc --warmup

COPY . .

RUN uv pip install --system --no-cache . && \
     uv pip install --system --no-cache -U "babeldoc<0.3.0" "pymupdf<1.25.3" "pdfminer-six==20250416" && \
     mkdir -p /data /root/.config/PDFMathTranslate /app/pdf2zh_files && \
     babeldoc --version && \
     babeldoc --warmup

VOLUME ["/data", "/root/.config/PDFMathTranslate"]

CMD ["pdf2zh", "-i", "--serverport", "7860"]
