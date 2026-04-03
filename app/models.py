from pydantic import BaseModel, Field


class TranslationPair(BaseModel):
    source_language: str
    target_language: str
    sentence: str
    translation: str

class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    query_sentence: str

class StammeringRequest(BaseModel):
    source_sentence: str
    translated_sentence: str

class PromptRequest(BaseModel):
    source_language: str = Field(..., description="ISO 639-1 source language code")
    target_language: str = Field(..., description="ISO 639-1 target language code")
    query_sentence: str = Field(..., description="Sentence to translate")

class StammeringResponse(BaseModel):
    has_stammer: bool

class AddPairResponse(BaseModel):
    status: str = "ok"
 
class PromptResponse(BaseModel):
    prompt: str

class HealthCheck(BaseModel):
    status: int