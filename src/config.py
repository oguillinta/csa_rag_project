import os

OPENAI_MODEL_CHAT = "gpt-4o-mini"
OPENAI_MODEL_EMBED = "text-embedding-3-small"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION_NAME = "support_kb"

# Scope/message for role
ROLE_NAME = "Customer Support Agent"
ROLE_SCOPE = (
    "order status, cancellations, returns, shipping dates, delivery issues, and basic customer/order details"
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "4"))
