# main.py
# A simple FastAPI example to test the server and authentication.
# This version returns a placeholder "dummy" response.

# --- Dependencies ---
# You'll only need fastapi and uvicorn for this simple example:
# pip install "fastapi[all]" uvicorn

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List

# --- Configuration ---

# The secret token for authentication.
API_KEY = "f5b00b57698f28dbb878319109a149d5cd0a7d430c25cf3c590b30d31cf8b028"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple Test API",
    description="A basic API to demonstrate endpoint setup and authentication.",
    version="1.0.0"
)

# --- Authentication ---
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """A dependency function to verify the bearer token."""
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models for Data Validation ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]


# --- API Endpoint ---
@app.post('/hackrx/run', response_model=HackRxResponse)
async def run_hackrx_simple(payload: HackRxRequest, token: str = Depends(verify_token)):
    """
    A simple endpoint that receives the request and returns a dummy response.
    It doesn't process the PDF, it just confirms it received the questions.
    """
    print("Request received successfully!")
    print(f"Document URL: {payload.documents}")
    print(f"Received {len(payload.questions)} questions.")

    # Create a simple list of placeholder answers.
    # The number of answers will match the number of questions received.
    dummy_answers = [f"This is a placeholder answer for question #{i+1}" for i, q in enumerate(payload.questions)]

    # Return the dummy data in the required format.
    return HackRxResponse(answers=dummy_answers)


# --- How to Run This Application ---
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)

