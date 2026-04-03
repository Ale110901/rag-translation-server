from http.client import HTTPException
import uvicorn

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging

from app.database import VectorDatabase
from app.stammering import StammeringDetector
from app.models import HealthCheck, PromptRequest, TranslationPair, StammeringRequest, \
                       PromptResponse, StammeringResponse, AddPairResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Translation Server",
    description="Retrieval-Augmented Generation backend for translation prompts",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDatabase()
stammering_detector = StammeringDetector()


#  POST /pairs  – store a new translation pair
@app.post("/pairs", response_model=AddPairResponse, status_code=201)
def add_pair(pair: TranslationPair):
    try:
        if not pair.source_language or not pair.target_language:
            raise HTTPException(status_code=422, detail="source_language and target_language are required.")

        if not pair.sentence or not pair.translation:
            raise HTTPException(status_code=422, detail="sentence and translation are required.")

        db.add_pair(pair)

        logger.info(
            "Stored pair [%s→%s]: %s",
            pair.source_language,
            pair.target_language,
            pair.sentence[:60],
        )

        return AddPairResponse(status="Ok")

    except Exception as e:
        logger.exception("Failed to add translation pair")
        raise HTTPException(status_code=500, detail=str(e))


#  GET /prompt  – retrieve RAG translation prompt
@app.get("/prompt", response_model=PromptResponse, status_code=201)
def get_prompt(request: PromptRequest = Depends()):
    try:
        similar_pairs = db.search(source_language=request.source_language,
                                  target_language=request.target_language,
                                  query=request.query_sentence,
                                  top_k=4)

        prompt = build_prompt(src=request.source_language,
                              tgt=request.target_language,
                              query_sentence=request.query_sentence,
                              examples=similar_pairs)

        logger.info("Prompt built with %d examples for: %s",
                    len(similar_pairs),
                    request.query_sentence[:60])

        return PromptResponse(prompt=prompt)

    except Exception as e:
        logger.exception("Failed to build prompt")
        raise HTTPException(status_code=500, detail=str(e))


#  GET /stammering  – detect translation stammering
@app.get("/stammering", response_model=StammeringResponse, status_code=201)
def detect_stammering(request: StammeringRequest = Depends()):
    """Detect stammering in a translated sentence."""

    has_stammer = stammering_detector.detect(request.source_sentence,
                                             request.translated_sentence)
    logger.info("Stammering check -> %s", has_stammer)
    return StammeringResponse(has_stammer=has_stammer)


#  Prompt builder
def build_prompt(src: str,
                 tgt: str,
                 query_sentence: str,
                 examples: list[dict]) -> str:
    """Construct a RAG prompt for an LLM translator (few shot prompt builder)."""

    lang_names = {
        "en": "English", "it": "Italian", "fr": "French", "de": "German",
        "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
        "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
        "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
        "tr": "Turkish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    }
    src_name = lang_names.get(src.lower(), src.upper())
    tgt_name = lang_names.get(tgt.lower(), tgt.upper())

    lines = [f"You are a professional translator specializing in {src_name} to {tgt_name} translation.",
             f"Translate the following sentence from {src_name} to {tgt_name}."]

    if examples:
        lines.append(f"Here are {len(examples)} similar translation example(s) to guide your translation:")
        lines.append("")
        for i, ex in enumerate(examples, 1):
            score = ex.get("score", 0.0)
            lines.append(f"Example {i} (similarity: {score:.2f}):")
            lines.append(f"  {src_name}: {ex['sentence']}")
            lines.append(f"  {tgt_name}: {ex['translation']}")
    else:
        logger.info("No similar examples found in the database for the current request.")

    lines += [
        "Now translate the following sentence:",
        f"  {src_name}: {query_sentence}",
        f"  {tgt_name}:",
    ]
    return "\n".join(lines)


#  Health check
@app.get("/health")
def health():
    return JSONResponse(status_cose=200,
                        content=HealthCheck(status=db.count()))

def start(host: str = "0.0.0.0", 
          port: int = 8000):
    # start uvicorn
    uvicorn.run("main:app", host=host, port=port)

if __name__ == "__main__":
    start()