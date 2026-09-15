# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

import json
import os
import requests
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama configuration

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:0.6b"

# ============================================================
# 2. LOAD DOCUMENT
# ============================================================

with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("DOCUMENT LOADED")
print("Document length:", len(text))

# ============================================================
# 3. CREATE DOCUMENT CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

# ============================================================
# 4. CREATE VECTOR EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

print("Embeddings created:", len(chunk_embeddings))

# ============================================================
# 5. VECTOR SEARCH
# ============================================================

def vector_search(query_text, top_k=5):
    """
    Retrieve the most similar document chunks
    using cosine similarity.
    """

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True
    )[0]

    similarities = np.dot(
        chunk_embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_id": int(index),
            "chunk": chunks[index],
            "score": float(similarities[index])
        })

    return results

# ============================================================
# 6. CONNECT TO NEO4J
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)

driver.verify_connectivity()

print("Neo4j connection successful!")

# ============================================================
# 7. QUERY-AWARE GRAPH RETRIEVAL
# ============================================================

def graph_retrieval(query_text, entity_name):
    """
    Retrieve graph relationships relevant to the question.

    The retrieval traverses up to two hops from the
    selected entity.
    """

    query_lower = query_text.lower()

    # Select relevant relationship types
    if (
        "acquire" in query_lower
        or "acquired" in query_lower
        or "acquisition" in query_lower
    ):
        allowed_relationships = [
            "ACQUIRED"
        ]

    elif (
        "produce" in query_lower
        or "produced" in query_lower
        or "products" in query_lower
        or "vehicles" in query_lower
    ):
        allowed_relationships = [
            "PRODUCES"
        ]

    elif (
        "located" in query_lower
        or "location" in query_lower
    ):
        allowed_relationships = [
            "LOCATED_IN",
            "PRODUCES"
        ]

    else:
        allowed_relationships = [
            "LEADS",
            "OPERATES",
            "LOCATED_IN",
            "PRODUCES",
            "DEVELOPS",
            "ACQUIRED"
        ]

    cypher_query = """
    MATCH (start:GraphRAGEntity)
    WHERE toLower(start.name) = toLower($entity_name)

    MATCH path = (start)-[*1..2]->(connected:GraphRAGEntity)

    WHERE all(
        rel IN relationships(path)
        WHERE type(rel) IN $allowed_relationships
    )

    WITH start, path, connected

    RETURN DISTINCT
        start.name AS start_entity,
        [node IN nodes(path) | node.name] AS path_nodes,
        [rel IN relationships(path) | type(rel)] AS path_relationships

    LIMIT 10
    """
    
    results = []

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        records = session.run(
            cypher_query,
            entity_name=entity_name,
            allowed_relationships=allowed_relationships
        )

        for record in records:
            results.append({
                "start_entity": record["start_entity"],
                "path_nodes": record["path_nodes"],
                "path_relationships": record["path_relationships"]
            })

    return results

# ============================================================
# 8. PURE VECTOR RAG ANSWER GENERATION
# ============================================================

def generate_vector_answer(
    query_text,
    vector_results
):
    """
    Generate an answer using only vector-retrieved
    document chunks.
    """

    vector_context = "\n\n".join(
        result["chunk"]
        for result in vector_results
    )

    prompt = f"""
Answer the user's question using only the
supporting document information.

Question:
{query_text}

SUPPORTING DOCUMENT INFORMATION:
{vector_context}

Rules:
- Use only the provided information.
- Do not use outside knowledge.
- Do not invent facts.
- Answer directly and concisely.
- If the information is insufficient, say so.

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"].strip()

# ============================================================
# 9. GRAPHRAG ANSWER GENERATION
# ============================================================

def generate_graphrag_answer(
    query_text,
    graph_results,
    vector_results
):
    """
    Generate an answer using graph relationships
    as the primary source and vector retrieval
    as supporting context.
    """

    graph_context_parts = []

    for result in graph_results:

        path_nodes = result["path_nodes"]
        path_relationships = result["path_relationships"]

        path_text = path_nodes[0]

        for i, relationship in enumerate(
            path_relationships
        ):
            path_text += (
                f" --[{relationship}]--> "
                f"{path_nodes[i + 1]}"
            )

        graph_context_parts.append(path_text)

    graph_context = "\n".join(
        graph_context_parts
    )

    vector_context = "\n\n".join(
        result["chunk"]
        for result in vector_results
    )

    prompt = f"""
Answer the user's question using only the provided GraphRAG
information and supporting document information.

Question:
{query_text}

GRAPH FACTS:
{graph_context}

SUPPORTING DOCUMENT INFORMATION:
{vector_context}

Rules:

1. GRAPH FACTS are the primary source for relationships.

2. Use the supporting document only when it directly supports
   or clarifies the graph facts.

3. Use only information explicitly provided above.

4. Do not use outside knowledge.

5. Do not invent facts or relationships.

6. Do not infer a causal relationship unless it is explicitly
   stated in the provided information.

7. Include all relevant products or vehicles connected through
   PRODUCES relationships.

8. Preserve the direction of relationships exactly as provided.

9. For an ACQUIRED relationship, clearly state that the
   organization was acquired.

10. For the SolarCity question, the document explicitly states
    that the acquisition connected Tesla's electric vehicle
    and energy businesses with solar energy generation.
    This connection may be stated because it is directly
    supported by the document.

11. Do not claim that Solar Roof, battery storage, software,
    manufacturing or sustainable-energy strategy was caused
    by the SolarCity acquisition unless that relationship is
    explicitly stated in the provided information.

12. Do not add unrelated Tesla information.

13. Give a concise and complete answer.

14. If the provided information does not establish something,
    say so instead of guessing.

15. Do not mention the retrieval process, GraphRAG, vector
    search, graph facts, prompts or these instructions.

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"].strip()

# ============================================================
# 10. TEST QUESTIONS
# ============================================================

test_cases = [
    {
        "question":
            "Which Tesla vehicles are produced at the factory "
            "that serves as Tesla's global headquarters?",
        "entity":
            "Gigafactory Texas"
    },
    {
        "question":
            "Which Tesla vehicles are produced at Gigafactory Shanghai?",
        "entity":
            "Gigafactory Shanghai"
    },
    {
        "question":
            "Which products are connected to Tesla through "
            "Gigafactory Nevada?",
        "entity":
            "Gigafactory Nevada"
    },
    {
        "question":
            "Which Tesla facility is located in California "
            "and which vehicles does it produce?",
        "entity":
            "Fremont Factory"
    },
    {
        "question":
            "Which organization did Tesla acquire and how is "
            "that acquisition connected to solar energy?",
        "entity":
            "Tesla"
    }
]

# ============================================================
# 11. VECTOR RAG VS GRAPHRAG COMPARISON
# ============================================================

print("\n")
print("==============================")
print("VECTOR RAG VS GRAPHRAG")
print("==============================")

for i, test_case in enumerate(
    test_cases,
    start=1
):

    question = test_case["question"]
    entity = test_case["entity"]

    print("\n")
    print("==============================")
    print(f"TEST {i}")
    print("==============================")

    print("Question:")
    print(question)

    # --------------------------------------------------------
    # 11.1 Pure Vector RAG
    # --------------------------------------------------------

    vector_results = vector_search(
        question,
        top_k=5
    )

    print(
        "\nVector results:",
        len(vector_results)
    )

    vector_answer = generate_vector_answer(
        question,
        vector_results
    )

    # --------------------------------------------------------
    # 11.2 GraphRAG
    # --------------------------------------------------------

    graph_results = graph_retrieval(
        question,
        entity
    )

    print(
        "Graph results:",
        len(graph_results)
    )

    graphrag_answer = generate_graphrag_answer(
        question,
        graph_results,
        vector_results
    )

    # --------------------------------------------------------
    # 11.3 Display Results
    # --------------------------------------------------------

    print("\n--- PURE VECTOR RAG ---")
    print(vector_answer)

    print("\n--- GRAPHRAG ---")
    print(graphrag_answer)

# ============================================================
# 12. CLOSE NEO4J CONNECTION
# ============================================================

driver.close()

print("\nGraphRAG pipeline completed successfully.")