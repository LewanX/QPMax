import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx

app = FastAPI()

class Query(BaseModel):
    prompt: str
    model: str = "llama2"
    stream: Optional[bool] = True  # Default to streaming


# Async generator for extracting "response" from /api/generate (Streaming)
async def stream_generated_text(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST", url, json={"model": model, "prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to connect to the model server."
                    )

                # Yield only the raw "response" content from the JSON objects
                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode('utf-8')
                    for line in decoded_chunk.split("\n\n"):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# Helper function for non-streaming response from /api/generate
async def get_generated_text(prompt: str, model: str):
    url = "http://localhost:11434/api/generate"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"model": model, "prompt": prompt})
            response.raise_for_status()

            # Combine all responses into a single string
            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

            return {"response": combined_response}

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")

# Async generator for extracting "response" from /api/chat (Streaming)

# Endpoint for /generate that handles both streaming and non-streaming
@app.post("/api/generate")
async def generate_text(query: Query):
    if query.stream:
        return StreamingResponse(
            stream_generated_text(query.prompt, query.model),
            media_type="text/plain"
        )
    else:
        response = await get_generated_text(query.prompt, query.model)
        return JSONResponse(response)

# Endpoint for /chat that handles both streaming and non-streaming
@app.post("/api/models/download")
async def download_model(llm_name: str = Body(..., embed=True)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/pull",
                json={"name": llm_name}
            )
            response.raise_for_status()
            return {"message": f"Model {llm_name} downloaded successfully"}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            return {"models": response.json()["models"]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)