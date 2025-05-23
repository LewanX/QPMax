import json
import os
import shutil
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
from pathlib import Path

# Importar el servicio RAG
from rag_service import get_rag_service

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str
    model: str = "llama2"
    stream: Optional[bool] = True  # Default to streaming

class RAGQuery(BaseModel):
    query: str
    model: str = "llama2"
    use_rag: bool = True

# Directorio temporal para almacenar archivos subidos
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Directorio para la base de conocimientos
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)

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

# Endpoint para consultar la base de conocimiento con RAG
@app.post("/api/rag/query")
async def rag_query(query: RAGQuery):
    """
    Consulta la base de conocimientos utilizando RAG y devuelve una respuesta enriquecida.
    
    Si use_rag es verdadero, consulta primero la base de conocimientos y luego usa el modelo.
    Si use_rag es falso, usa directamente el modelo sin RAG.
    """
    try:
        if query.use_rag:
            # Usar el servicio RAG para obtener una respuesta basada en la base de conocimientos
            rag_service = get_rag_service()
            result = await rag_service.query(query.query, model=query.model)
            
            if result["success"]:
                return JSONResponse(result)
            else:
                # Si RAG falla, intentamos con el modelo directo
                response = await get_generated_text(query.query, query.model)
                result = {
                    "success": True,
                    "query": query.query,
                    "response": response["response"],
                    "has_sources": False,
                    "sources": [],
                    "fallback": True
                }
                return JSONResponse(result)
        else:
            # Usar directamente el modelo sin RAG
            response = await get_generated_text(query.query, query.model)
            return JSONResponse({
                "success": True,
                "query": query.query,
                "response": response["response"],
                "has_sources": False,
                "sources": []
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error procesando la consulta: {str(e)}",
            "query": query.query,
            "response": "Se produjo un error al procesar su consulta."
        }, status_code=500)

# Función para generar un stream de error
async def create_error_stream(error_message: str):
    """Helper function para crear un stream de error."""
    yield f"Error: {error_message}"

# Nuevo endpoint para streaming RAG
@app.post("/api/rag/stream")
async def stream_rag_query(query: RAGQuery):
    """
    Consulta la base de conocimientos utilizando RAG y devuelve una respuesta en streaming.
    
    Si use_rag es verdadero, consulta la base de conocimientos con streaming.
    Si use_rag es falso, usa directamente el modelo con streaming.
    """
    try:
        if query.use_rag:
            # Usar el servicio RAG en modo streaming
            rag_service = get_rag_service()
            return StreamingResponse(
                rag_service.stream_query(query.query, model=query.model),
                media_type="text/plain"
            )
        else:
            # Usar directamente el modelo con streaming
            return StreamingResponse(
                stream_generated_text(query.query, query.model),
                media_type="text/plain"
            )
    except Exception as e:
        # Usar la función helper con el mensaje de error
        error_message = str(e)
        return StreamingResponse(
            create_error_stream(error_message),
            media_type="text/plain"
        )

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

# Endpoint para subir documentos a la base de conocimientos
@app.post("/api/rag/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Sube uno o más documentos a la base de conocimientos.
    Soporta formatos de texto (.txt, .md), PDF (.pdf), y documentos (.docx, .doc).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
    
    saved_files = []
    temp_files = []
    
    try:
        # Guardar archivos temporalmente
        for file in files:
            # Verificar extensión de archivo
            file_ext = Path(file.filename).suffix.lower()
            supported_extensions = [".txt", ".pdf", ".docx", ".doc", ".md", ".csv", ".json"]
            
            if file_ext not in supported_extensions:
                continue  # Saltarse archivos no soportados
            
            # Guardar el archivo temporalmente
            temp_file_path = TEMP_DIR / file.filename
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Copiar a la base de conocimientos
            target_path = KNOWLEDGE_BASE_DIR / file.filename
            shutil.copy(temp_file_path, target_path)
            
            saved_files.append(str(target_path))
            temp_files.append(str(temp_file_path))
        
        if not saved_files:
            return JSONResponse({
                "success": False,
                "message": "No se subieron archivos válidos. Formatos soportados: TXT, PDF, DOCX, DOC, MD, CSV, JSON"
            })
        
        # Actualizar el índice con los nuevos documentos
        rag_service = get_rag_service()
        result = await rag_service.add_documents(saved_files)
        
        return JSONResponse({
            "success": True,
            "message": f"Se subieron {len(saved_files)} documentos a la base de conocimientos",
            "files": [Path(f).name for f in saved_files],
            "index_update": result
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error al procesar los archivos: {str(e)}"
        }, status_code=500)
    
    finally:
        # Limpiar archivos temporales
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Endpoint para listar documentos en la base de conocimientos
@app.get("/api/rag/documents")
async def list_documents():
    """Lista todos los documentos disponibles en la base de conocimientos."""
    try:
        documents = []
        for file_path in KNOWLEDGE_BASE_DIR.iterdir():
            if file_path.is_file():
                documents.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime
                })
        
        return JSONResponse({
            "success": True,
            "count": len(documents),
            "documents": documents
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error al listar documentos: {str(e)}"
        }, status_code=500)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)