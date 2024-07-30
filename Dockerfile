FROM python:3.12.4-slim-bookworm

RUN pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]

ENTRYPOINT ["optimum-cli"]
CMD ["--help"]
