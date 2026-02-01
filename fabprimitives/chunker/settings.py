from pydantic import BaseModel, Field, validator
from typing import Optional

class ChunkerSettings(BaseModel):
    chunk_max_length: int = Field(2000, description="Maximum length of each chunk in characters (1024-30000)")
    chunk_overlap: int = Field(256, description="Character overlap between consecutive chunks (≥256)")
    
    @validator('chunk_max_length')
    def validate_chunk_max_length(cls, v):
        if v < 100 or v > 30000:  # Reduced minimum to allow smaller chunks
            raise ValueError('chunk_max_length must be between 100 and 30000 characters')
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if v < 50:  # Reduced minimum to allow smaller overlap
            raise ValueError('chunk_overlap must be at least 50 characters')
        if 'chunk_max_length' in values and v >= values['chunk_max_length']:
            raise ValueError('chunk_overlap must be less than chunk_max_length')
        return v
