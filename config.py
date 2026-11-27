"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI


def is_elevenlabs_key_id(value: str) -> bool:
    """Return True if value matches ElevenLabs' 64-hex API Key ID format instead of an 'sk_' secret key."""
    if not value or not isinstance(value, str):
        return False
    cleaned = value.strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", cleaned)) and not cleaned.startswith("sk_")

# Load environment variables from .env file
# Ensure .env is resolved reliably regardless of current working directory
_config_dir = os.path.dirname(os.path.abspath(__file__))
_local_env = os.path.join(_config_dir, ".env")
_parent_env = os.path.join(os.path.dirname(_config_dir), ".env")

if os.path.isfile(_local_env):
    load_dotenv(dotenv_path=_local_env)
elif os.path.isfile(_parent_env):
    load_dotenv(dotenv_path=_parent_env)
else:
    load_dotenv()

# Import provider adapter for Groq and LLM providers
try:
    from provider_config_adapter import get_llm_provider, get_groq_provider, LLM_PROVIDER_FALLBACK_ENABLED
except ImportError as e:
    raise ImportError(
        f"Failed to import provider configuration adapter: {e}. "
        f"Make sure groq_llm_provider.py and provider_config_adapter.py are present."
    )

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = get_groq_provider(
            temperature=0.1,  # Deterministic
            max_tokens=300,   # Tiny JSON decision output; prevents 8K token reservation
        )

class ConversationConfig:
    def __init__(self):
        self.llm = get_groq_provider(
            temperature=0.7,  # Creative but factual
            max_tokens=600,   # Generous for dialogue while fitting comfortably within Groq daily quotas
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = get_groq_provider(
            temperature=0.3,  # Slightly creative but factual
            max_tokens=600,
        )
        self.context_limit = 20     # include last 20 messages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Embedding Configuration: Hugging Face only
        # Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
        self.embedding_dim = 384
        self.collection_name = "medical_assistance_rag_hf"
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except ImportError as e:
            raise ImportError(
                "HuggingFace embeddings initialization failed: 'langchain_huggingface' package not found. "
                "Install it with: pip install langchain-huggingface"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize HuggingFace embeddings (model: sentence-transformers/all-MiniLM-L6-v2, device: cpu): {e}"
            ) from e
        self.llm = get_groq_provider(
            temperature=0.3,  # Slightly creative but factual
            max_tokens=600,
        )
        # Initialize Azure summarizer_model (for image text extraction) - lazy-loaded if vars are set
        deployment_name = os.getenv("deployment_name")
        if deployment_name:
            self.summarizer_model = AzureChatOpenAI(
                deployment_name = deployment_name,  # Replace with your Azure deployment name
                model_name = os.getenv("model_name"),  # Replace with your Azure model name
                azure_endpoint = os.getenv("azure_endpoint"),  # Replace with your Azure endpoint
                openai_api_key = os.getenv("openai_api_key"),  # Replace with your Azure OpenAI API key
                openai_api_version = os.getenv("openai_api_version"),  # Ensure this matches your API version
                temperature = 0.5  # Slightly creative but factual
            )
        else:
            self.summarizer_model = None
        self.chunker_model = get_groq_provider(
            temperature=0.0,  # factual
            max_tokens=500,
        )
        self.response_generator_model = get_groq_provider(
            temperature=0.3,  # Slightly creative but factual
            max_tokens=600,
        )
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class ClinicalConsultationConfig:
    def __init__(self):
        self.llm = get_groq_provider(
            temperature=0.3,  # Structured, factual, conversational
            max_tokens=700,
        )
        self.context_limit = 30  # Include more history for contextual understanding
        self.max_followup_questions = 8  # Maximum adaptive follow-up questions before mandatory confirmation
        self.emergency_keywords = ["chest pain", "difficulty breathing", "severe bleeding", "loss of consciousness"]
        self.patient_db_path = "./data/clinical_consultation_db"  # Path for patient data storage


class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        # Initialize Azure vision LLM (for image classification) - lazy-loaded if vars are set
        deployment_name = os.getenv("deployment_name")
        if deployment_name:
            self.llm = AzureChatOpenAI(
                deployment_name = deployment_name,  # Replace with your Azure deployment name
                model_name = os.getenv("model_name"),  # Replace with your Azure model name
                azure_endpoint = os.getenv("azure_endpoint"),  # Replace with your Azure endpoint
                openai_api_key = os.getenv("openai_api_key"),  # Replace with your Azure OpenAI API key
                openai_api_version = os.getenv("openai_api_version"),  # Ensure this matches your API version
                temperature = 0.1  # Keep deterministic for classification tasks
            )
        else:
            self.llm = None

class SpeechConfig:
    def __init__(self):
        raw_key = os.getenv("ELEVEN_LABS_API_KEY", "")
        self.eleven_labs_api_key = raw_key.strip() if raw_key else ""
        self.eleven_labs_voice_id = "EXAVITQu4vr4xnSDxMaL"  # Default premade voice (Sarah)
        if is_elevenlabs_key_id(self.eleven_labs_api_key):
            print(
                "Warning: ELEVEN_LABS_API_KEY appears to be a 64-char ElevenLabs API Key ID "
                "rather than an API secret key (expected prefix 'sk_'). STT and TTS requests will fail authentication."
            )

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.clinical_consultation = ClinicalConsultationConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        raw_key = os.getenv("ELEVEN_LABS_API_KEY", "")
        self.eleven_labs_api_key = raw_key.strip() if raw_key else ""
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()