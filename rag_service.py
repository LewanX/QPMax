import os
import logging  # Agregar logging
from typing import List, Dict, Optional, AsyncGenerator
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, docs_dir: str = "knowledge_base", persist_dir: str = "chroma_db"):
        """
        Inicializa el servicio RAG con ChromaDB.
        
        Args:
            docs_dir: Directorio donde se almacenan los documentos.
            persist_dir: Directorio donde se persiste ChromaDB.
        """
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)
        self.index = None
        self.chroma_client = None
        self.collection = None
        
        # Crear directorios si no existen
        self.docs_dir.mkdir(exist_ok=True, parents=True)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        
        # Configurar LLM y embedding
        Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
        # Usar embeddings de HuggingFace (gratuito y local)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Inicializar ChromaDB y el índice
        self._init_chroma_and_index()
        
    def _init_chroma_and_index(self):
        """Inicializa ChromaDB y el índice vectorial."""
        try:
            # Crear cliente ChromaDB persistente
            logger.info(f"Inicializando ChromaDB en {self.persist_dir}")
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Obtener o crear colección
            self.collection = self.chroma_client.get_or_create_collection("document_collection")
            
            # Crear vector store con la colección
            logger.info("Creando vector store con la colección ChromaDB")
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # Verificar si hay documentos para indexar
            if not any(self.docs_dir.iterdir()):
                logger.info("No hay documentos para indexar. El índice estará vacío.")
                # Crear un índice vacío usando el vector store
                self.index = VectorStoreIndex.from_vector_store(vector_store)
            else:
                # Cargar documentos e indexarlos
                logger.info(f"Cargando documentos desde {self.docs_dir}")
                documents = SimpleDirectoryReader(str(self.docs_dir)).load_data()
                logger.info(f"Creando índice con {len(documents)} documentos")
                self.index = self._create_index_with_advanced_settings(documents, vector_store)
                logger.info(f"Índice creado/actualizado con {len(documents)} documentos.")
                
        except Exception as e:
            error_msg = f"Error al inicializar ChromaDB y el índice: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.chroma_client = None
            self.collection = None
            self.index = None
            raise RuntimeError(error_msg) from e
    
    def _create_index_with_advanced_settings(self, documents, vector_store):
        """Crea un índice con configuraciones avanzadas para mejorar la recuperación."""
        from llama_index.core.node_parser import SimpleNodeParser
        
        # Configurar el parser de nodos con superposición
        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,     # Tamaño de chunks más pequeño para precisión
            chunk_overlap=50    # Superposición para mantener contexto
        )
        
        # Indexar con configuración personalizada
        index = VectorStoreIndex.from_documents(
            documents, 
            vector_store=vector_store,
            show_progress=True  # Muestra barra de progreso para ingesta grande
        )
        
        return index
    
    async def add_documents(self, file_paths: List[str]) -> Dict:
        """
        Añade documentos al índice.
        
        Args:
            file_paths: Lista de rutas a los archivos a añadir.
        
        Returns:
            Un diccionario con los resultados de la operación.
        """
        try:
            # Cargar nuevos documentos
            documents = SimpleDirectoryReader(input_files=file_paths).load_data()
            
            if not documents:
                return {"success": False, "message": "No se pudieron cargar documentos válidos"}
            
            # Si no existe el índice, crearlo
            if self.index is None:
                vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.index = self._create_index_with_advanced_settings(documents, vector_store)
            else:
                # Insertar nuevos documentos en el índice existente
                for doc in documents:
                    self.index.insert(doc)
            
            return {
                "success": True, 
                "message": f"Se agregaron {len(documents)} documentos al índice",
                "documents": [doc.get_content()[:100] + "..." for doc in documents]
            }
        except Exception as e:
            return {"success": False, "message": f"Error al añadir documentos: {str(e)}"}
    
    async def query(self, query_text: str, model: str = "llama2") -> Dict:
        """
        Consulta la base de conocimientos.
        
        Args:
            query_text: Texto de la consulta.
            model: Modelo a utilizar para la respuesta (por defecto llama2).
            
        Returns:
            Un diccionario con los resultados y contexto de la consulta.
        """
        if not self.index:
            logger.error("Índice no disponible para consulta")
            return {
                "success": False,
                "message": "El índice no está disponible",
                "response": "No se pudo acceder a la base de conocimientos."
            }
        
        try:
            logger.info(f"Iniciando consulta con modelo: {model}")
            # Configurar el LLM específico para esta consulta
            custom_llm = Ollama(model=model, request_timeout=120.0)
            
            # Configurar el retriever para obtener los fragmentos más relevantes
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,
            )
            
            # Preparar el motor de consulta
            logger.info("Preparando response synthesizer y query engine")
            
            # Configurar temporalmente el LLM para esta consulta
            from llama_index.core import Settings as LlamaSettings
            original_llm = LlamaSettings.llm
            LlamaSettings.llm = custom_llm
            
            try:
                response_synthesizer = get_response_synthesizer()
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer
                )
                
                # Realizar la consulta
                logger.info(f"Ejecutando consulta: {query_text}")
                response = query_engine.query(query_text)
            finally:
                # Restaurar el LLM original
                LlamaSettings.llm = original_llm
            
            # Extraer los nodos de origen (contexto utilizado)
            logger.info("Procesando nodos de origen")
            source_nodes = response.source_nodes
            sources = []
            
            for node in source_nodes:
                sources.append({
                    "text": node.node.get_content()[:200] + "...",
                    "score": node.score if hasattr(node, "score") else None,
                    "doc_id": node.node.doc_id if hasattr(node.node, "doc_id") else None,
                    "metadata": node.node.metadata if hasattr(node.node, "metadata") else {}
                })
            
            logger.info(f"Consulta completada con éxito. Fuentes encontradas: {len(sources)}")
            return {
                "success": True,
                "query": query_text,
                "response": str(response),
                "has_sources": len(sources) > 0,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error al consultar la base de conocimientos: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error al consultar la base de conocimientos: {str(e)}",
                "response": "Se produjo un error al procesar su consulta."
            }

    async def stream_query(self, query_text: str, model: str = "llama2") -> AsyncGenerator[str, None]:
        """
        Consulta la base de conocimientos con streaming de respuesta.
        
        Args:
            query_text: Texto de la consulta.
            model: Modelo a utilizar para la respuesta (por defecto llama2).
            
        Yields:
            Fragmentos de texto de la respuesta generada.
        """
        if not self.index:
            logger.error("Índice no disponible para consulta streaming")
            yield "Error: El índice no está disponible. No se pudo acceder a la base de conocimientos."
            return
        
        try:
            logger.info(f"Iniciando consulta streaming con modelo: {model}")
            # Configurar el LLM específico para esta consulta con streaming
            custom_llm = Ollama(model=model, request_timeout=120.0)
            
            # Configurar el retriever para obtener los fragmentos más relevantes
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,
            )
            
            # Extraer documentos relevantes
            logger.info(f"Recuperando documentos relevantes para: {query_text}")
            retrieved_nodes = retriever.retrieve(query_text)
            
            if not retrieved_nodes:
                logger.warning("No se encontraron documentos relevantes para la consulta")
                yield "No se encontraron documentos relevantes para tu consulta."
                return
                
            # Construir el contexto para el prompt
            logger.info(f"Construyendo contexto con {len(retrieved_nodes)} documentos")
            context_text = "\n\n".join([node.node.get_content() for node in retrieved_nodes])
            
            # Crear el prompt para Ollama con el contexto
            prompt = f"""Responde a la siguiente consulta basándote EXCLUSIVAMENTE en el contexto proporcionado.
            
Contexto:
{context_text}

Consulta: {query_text}

Respuesta:"""
            
            # URL para Ollama
            import httpx
            url = "http://localhost:11434/api/generate"
            
            # Hacer la solicitud con streaming
            logger.info("Iniciando streaming con Ollama")
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST", url, json={"model": model, "prompt": prompt}
                    ) as response:
                        if response.status_code != 200:
                            error_msg = f"Error al conectar con el servidor del modelo. Código: {response.status_code}"
                            logger.error(error_msg)
                            yield error_msg
                            return
                            
                        # Stream de la respuesta
                        logger.info("Recibiendo y procesando stream de respuesta")
                        async for chunk in response.aiter_bytes():
                            decoded_chunk = chunk.decode('utf-8')
                            for line in decoded_chunk.split("\n\n"):
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        if "response" in data:
                                            yield data["response"]
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Error decodificando JSON: {str(e)}")
                                        continue  # Skip invalid JSON
                logger.info("Streaming completado con éxito")
            except httpx.RequestError as e:
                error_msg = f"Error de conexión con Ollama: {str(e)}"
                logger.error(error_msg)
                yield error_msg
        
        except Exception as e:
            error_msg = f"Error al procesar la consulta streaming: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

# Importación para JSON
import json

# Instancia global del servicio RAG
rag_service = None

def get_rag_service():
    """Devuelve la instancia del servicio RAG, inicializándola si es necesario."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service 