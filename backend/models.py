from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True)
    english_label = Column(String, index=True)   
    nepali_text = Column(String)                 
    confidence = Column(Float)                   
    timestamp = Column(DateTime, default=datetime.utcnow)