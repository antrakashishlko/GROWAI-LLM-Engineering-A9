# GROWAI LLM Engineering - Assignment 9

## GraphRAG Knowledge Explorer with Neo4j

This project demonstrates a GraphRAG system that combines a **Neo4j knowledge graph** with **vector search** to answer questions involving connected facts and multi-hop reasoning.

A Tesla knowledge document is processed using the **Qwen3 0.6B** model through Ollama to extract entities and relationships. The extracted knowledge is stored in Neo4j and combined with vector retrieval using **Reciprocal Rank Fusion (RRF)**.

## Features

* Document chunking
* LLM-based entity extraction
* LLM-based relationship extraction
* Knowledge graph construction using Neo4j
* Graph-based retrieval
* Vector similarity search
* Hybrid GraphRAG retrieval
* Reciprocal Rank Fusion (RRF)
* Multi-hop question answering
* Pure Vector RAG vs GraphRAG comparison
* Graph validation and cleaning
* Edge-case handling during extraction
  
## Technologies Used

* Python
* Ollama
* Qwen3 0.6B
* Neo4j
* LangChain
* LangChain Core
* LangChain Ollama
* LangChain Text Splitters
* Neo4j Python Driver
* LangChain Neo4j
* Sentence Transformers
* NumPy
* python-dotenv

## Requirements

* Python 3.x
* Ollama
* Qwen3 0.6B
* Neo4j Aura or Neo4j Community
* Dependencies listed in `requirements.txt`

## Setup / Installation

Install the required dependencies:
```text
pip install -r requirements.txt
```

Download the required Ollama model:
```text
ollama pull qwen3:0.6b
```

Configure the Neo4j connection in `.env`:
```text
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_neo4j_database
```

## How to Run

**1. Run the main GraphRAG pipeline:**

Execute:
```text
python graphrag.py
```

The program:

* Loads the Tesla knowledge document.
* Splits the document into chunks.
* Extracts entities and relationships using Qwen3 0.6B.
* Cleans and validates the extracted graph data.
* Stores the knowledge graph in Neo4j.
* Creates vector embeddings for the document chunks.
* Performs graph retrieval and vector retrieval.
* Combines the results using RRF.
* Generates GraphRAG answers for the test questions.

**2. Compare Pure Vector RAG with GraphRAG:**

Execute:
```text
python compare_rag.py
```

This script compares Pure Vector RAG and GraphRAG using five test questions.

## Knowledge Graph

The project uses **Tesla** as the knowledge domain.

The final graph contains **44 entities and 33 relationships**.

Example relationships:

```text
Elon Musk → LEADS → Tesla
Tesla → OPERATES → Gigafactory Texas
Gigafactory Texas → LOCATED_IN → Austin, Texas
Gigafactory Texas → PRODUCES → Model Y
Gigafactory Texas → PRODUCES → Cybertruck
Tesla → ACQUIRED → SolarCity
```

## Test Cases

The project evaluates GraphRAG using five questions:

1. Which Tesla vehicles are produced at the factory that serves as Tesla's global headquarters?
2. Which Tesla vehicles are produced at Gigafactory Shanghai?
3. Which products are produced at Gigafactory Nevada?
4. Which vehicles or products are produced at Fremont Factory in California?
5. Which organization did Tesla acquire in 2016, and how did the acquisition connect Tesla to solar energy?
   
The same questions are used to compare Pure Vector RAG and GraphRAG.

## Vector RAG vs GraphRAG

The comparison demonstrates that GraphRAG can provide more precise answers when relationships between entities are important.

For example, for the Tesla global headquarters question:

**Pure Vector RAG** produced a generic response, while **GraphRAG** correctly identified **Model Y and Cybertruck** as the vehicles produced at Gigafactory Texas.

## Real-World Relevance

GraphRAG is useful for applications involving interconnected information, such as:

* Research assistants
* Enterprise knowledge bases
* Customer-support systems
* Scientific research
* Business and product analysis

## Edge Case / Failure Point

A possible failure point is incorrect or incomplete entity extraction from a document chunk.

The project handles this by cleaning and validating extracted entities and relationships before storing them in Neo4j.

If a question does not match relevant graph or document information, the system can return no relevant information instead of generating an unsupported relationship.

Invalid JSON returned during LLM extraction is also handled so that invalid graph data is not inserted into Neo4j.

## Project Files

* `graphrag.py` – Main GraphRAG pipeline
* `compare_rag.py` – Vector RAG vs GraphRAG comparison
* `document.txt` – Tesla knowledge document
* `requirements.txt` – Project dependencies
* `.env` – Neo4j configuration
* `extracted_graph.json` – Extracted entities and relationships
* `.gitignore` – Version-control exclusions

## Assignment

GROWAI LLM Engineering & Generative AI – Assignment 9

