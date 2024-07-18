from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import boto3
from api_keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def get_embedding_function():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="ap-southeast-2"  # Make sure this matches your Bedrock region
    )

    client = session.client('bedrock-runtime')

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-image-v1",
        client=client,
    )

    return embeddings