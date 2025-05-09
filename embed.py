tokenizer = None
model = None

import os
# Make sure directory exist
os.makedirs('/tmp', exist_ok=True)
#  Set Hugging Face cache directories to Lambda writable dir
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp'




import io
import json
import boto3
import faiss
import torch
import tempfile
import tarfile
from urllib.parse import unquote_plus
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader

s3 = boto3.client('s3')

BUCKET = os.environ['BUCKET_NAME']
INDEX_FILE = os.environ['INDEX_FILE']
METADATA_FILE = os.environ['METADATA_FILE']
MODEL_FILE = os.environ['MODEL_FILE']  # e.g., 'models/all-MiniLM-L6-v2.tar.gz'

# Lazy-load model for reuse across Lambda invocations
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        model_dir = "/tmp/model"
        model_tar = "/tmp/model.tar.gz"

        # Download and extract model if not already present
        if not os.path.exists(model_dir):
            print(f"Downloading model from S3: {MODEL_FILE}")
            s3.download_file(BUCKET, MODEL_FILE, model_tar)
            with tarfile.open(model_tar, "r:gz") as tar:
                tar.extractall(model_dir)
        #  Set to the actual model folder inside the extracted tar
        model_dir = os.path.join(model_dir, "all-MiniLM-L12-v2") 
               
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        model.eval()
    return tokenizer, model

def extract_text_from_pdf(stream):
    reader = PdfReader(stream)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts):
    tokenizer, model = load_model()
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return embeddings.numpy()

def handler(event, context):
    key = unquote_plus(event['Records'][0]['s3']['object']['key'])
    print(f"Processing file: {key}")

    if not key.lower().endswith('.pdf'):
        return {"statusCode": 400, "body": f"Unsupported file type: {key}"}

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    body = io.BytesIO(obj['Body'].read())
    text = extract_text_from_pdf(body)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, INDEX_FILE)
        metadata_path = os.path.join(tmpdir, METADATA_FILE)

        try:
            s3.download_file(BUCKET, INDEX_FILE, index_path)
            index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print("Loaded existing FAISS index and metadata.")
        except Exception as e:
            print(f"Starting fresh index due to error: {e}")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            metadata = []

        index.add(embeddings)
        metadata.extend(chunks)

        faiss.write_index(index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        s3.upload_file(index_path, BUCKET, INDEX_FILE)
        s3.upload_file(metadata_path, BUCKET, METADATA_FILE)

    return {"statusCode": 200, "body": f"Indexed {len(chunks)} chunks from {key}"}

