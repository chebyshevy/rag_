FROM public.ecr.aws/lambda/python:3.9

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --target .

COPY embed.py .

CMD ["embed.handler"]

