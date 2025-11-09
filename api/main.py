from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class TextItem(BaseModel):
    """A processed text item"""
    type_item: str = Field(description="ONLY OPTIONS ARE 'note' or 'todo'")
    text: str = Field(description="The text of the item")


class TextResponse(BaseModel):
    """Response containing an array of text items"""
    items: List[TextItem] = Field(description="Array of processed text items")


class TextRequest(BaseModel):
    text: str



@app.post("/")
async def process_text(request: TextRequest) -> List[dict]:
    """
    Accepts text and returns a structured JSON array.
    Uses LangChain with OpenAI to process the text.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    structured_llm = llm.with_structured_output(TextResponse)
    
    prompt = f"You are a helpful assistant that uses the prmpts given by the user to summarize thoughts, ideas, and todos. Return the results in a structured JSON array. Break down the text into meaningful items: {request.text}"
    
    result = await structured_llm.ainvoke(prompt)
    
    # Convert to list of dicts for JSON response
    return [item.dict() for item in result.items]