FROM python:3.12-slim
RUN pip install --no-cache-dir flask requests
WORKDIR /app
COPY compatibility/client-sdk/ /opt/aihub-client-sdk/
RUN pip install --no-cache-dir /opt/aihub-client-sdk
COPY app.py gpu_runner.py genre_packs.json /app/
CMD ["python", "/app/app.py"]
