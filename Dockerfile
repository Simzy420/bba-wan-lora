FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /opt/bba
COPY requirements.txt /opt/bba/requirements.txt
# The Runpod base image ships Debian cryptography without a pip RECORD file.
RUN pip install --no-cache-dir --ignore-installed -r /opt/bba/requirements.txt
COPY entrypoint.py bootstrap.py handler.py train_wan22_lora.py /opt/bba/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Override the base image entrypoint so Jupyter does not start.
ENTRYPOINT ["python", "-u", "/opt/bba/entrypoint.py"]
