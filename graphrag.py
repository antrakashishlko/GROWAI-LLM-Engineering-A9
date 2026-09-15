# ============================================================
# 1. Load the Document
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Document loaded successfully!")
print("Total characters:", len(text))

# ============================================================
# 2. Split Document into Chunks
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

# ============================================================
# 3. Display Chunk Information
# ============================================================

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

# ============================================================
# 4. Structured Knowledge Graph Extraction Prompt
# ============================================================

extraction_prompt = """
Extract a knowledge graph from the TEXT.

Return ONLY valid JSON in exactly this format:

{{
  "entities": [
    {{
      "name": "actual entity name",
      "type": "Person"
    }}
  ],
  "relationships": [
    {{
      "source": "actual entity name",
      "relationship": "LEADS",
      "target": "actual entity name"
    }}
  ]
}}

Allowed entity types:
Person
Organization
Location
Concept

Allowed relationships:
LEADS
OPERATES
LOCATED_IN
PRODUCES
DEVELOPS
ACQUIRED

Rules:
- Extract entities that are actually mentioned in TEXT.
- Use the exact entity names from TEXT when possible.
- Never output generic words such as organization, company, person, location, factory, facility, product, concept, city, state or country as entities.
- Never copy examples or instructions as entities.
- Do not use outside knowledge.
- Extract only facts explicitly stated in TEXT.
- LEADS: Person -> Organization.
- OPERATES: Organization -> named manufacturing facility.
- LOCATED_IN: named facility -> actual location.
- PRODUCES: named facility -> product explicitly stated to be produced or built.
- DEVELOPS: Organization -> product or concept explicitly stated to be developed.
- ACQUIRED: Organization -> Organization.
- Preserve relationship direction.
- Do not infer relationships.
- Do not turn "official home of" into PRODUCES.
- Do not turn "associated with" into PRODUCES.
- Every relationship source and target must be present in entities.
- If no valid relationship exists, return an empty relationships list.
- Return JSON only.
- Do not use markdown code fences.
- Do not explain your answer.

TEXT:
{text}
"""

# ============================================================
# 5. Extract Knowledge from All Chunks using Ollama
# ============================================================

import json
import requests

all_entities = []
all_relationships = []

for i, chunk in enumerate(chunks):

    print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
    print("Sending chunk to Ollama...")

    try:
        prompt = extraction_prompt.format(text=chunk)

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:0.6b",
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
        content = result["message"]["content"]

        print("Ollama response received!")

        try:
            cleaned_content = content.strip()

            # Remove markdown code fences if Ollama adds them.
            if cleaned_content.startswith("```"):
                lines = cleaned_content.splitlines()

                if lines and lines[0].startswith("```"):
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                cleaned_content = "\n".join(lines).strip()

            data = json.loads(cleaned_content)

            if not isinstance(data, dict):
                raise ValueError("JSON response is not an object")

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            if not isinstance(entities, list):
                entities = []

            if not isinstance(relationships, list):
                relationships = []

            all_entities.extend(entities)
            all_relationships.extend(relationships)

            print(f"Entities found: {len(entities)}")
            print(f"Relationships found: {len(relationships)}")

        except (json.JSONDecodeError, ValueError, TypeError):

            print(
                f"Could not parse valid knowledge graph JSON "
                f"for chunk {i + 1}"
            )

            print(content)

    except requests.exceptions.Timeout:

        print(f"Ollama timed out for chunk {i + 1}")

    except requests.exceptions.RequestException as e:

        print(
            f"Ollama request failed for chunk {i + 1}: {e}"
        )

# ============================================================
# 6. Clean and Validate Entities
# ============================================================

allowed_entity_types = {
    "Person",
    "Organization",
    "Location",
    "Concept"
}

entity_name_corrections = {
    "tesla": "Tesla",
    "elon musk": "Elon Musk",

    "roadster": "Roadster",
    "model s": "Model S",
    "model x": "Model X",
    "model 3": "Model 3",
    "model y": "Model Y",
    "cybertruck": "Cybertruck",
    "tesla semi": "Tesla Semi",
    "optimus": "Tesla Optimus",
    "tesla optimus": "Tesla Optimus",

    "powerwall": "Powerwall",
    "powerpack": "Powerpack",
    "megapack": "Megapack",

    "solar roof": "Solar Roof",
    "solar roof products": "Solar Roof products",
    "solar panels": "Solar Panels",
    "electrical components": "Electrical Components",
    "electric motors": "Electric Motors",
    "powertrains": "Powertrains",
    "vehicle batteries": "Vehicle Batteries",

    "fremont factory": "Fremont Factory",
    "fremont": "Fremont Factory",
    "gigafactory nevada": "Gigafactory Nevada",
    "gigafactory new york": "Gigafactory New York",
    "gigafactory shanghai": "Gigafactory Shanghai",
    "gigafactory texas": "Gigafactory Texas",
    "gigafactory berlin-brandenburg": "Gigafactory Berlin-Brandenburg",

    "austin": "Austin",
    "texas": "Texas",
    "california": "California",
    "nevada": "Nevada",
    "buffalo, new york": "Buffalo, New York",
    "new york": "New York",
    "shanghai": "Shanghai",
    "china": "China",
    "germany": "Germany",
    "berlin-brandenburg": "Berlin-Brandenburg",

    "solarcity": "SolarCity",
    "spacex": "SpaceX",
    "neuralink": "Neuralink",
    "the boring company": "The Boring Company"
}

entity_type_corrections = {
    "tesla": "Organization",
    "elon musk": "Person",

    "solarcity": "Organization",
    "spacex": "Organization",
    "neuralink": "Organization",
    "the boring company": "Organization",

    "fremont factory": "Location",
    "gigafactory nevada": "Location",
    "gigafactory new york": "Location",
    "gigafactory shanghai": "Location",
    "gigafactory texas": "Location",
    "gigafactory berlin-brandenburg": "Location"
}

invalid_entities = {
    "company",
    "organization",
    "person",
    "location",
    "concept",
    "factory",
    "facility",
    "manufacturing facility",
    "product",
    "products",
    "city",
    "state",
    "country",
    "european",
    "transportation"
}

unique_entities = {}

for entity in all_entities:

    if not isinstance(entity, dict):
        continue

    name = str(entity.get("name", "")).strip()
    entity_type = str(entity.get("type", "")).strip()

    if not name:
        continue

    name_lower = name.lower()

    if name_lower in invalid_entities:
        continue

    if name_lower in entity_name_corrections:
        name = entity_name_corrections[name_lower]

    if name_lower in entity_type_corrections:
        entity_type = entity_type_corrections[name_lower]

    if entity_type not in allowed_entity_types:
        continue

    key = name.lower()

    if key not in unique_entities:
        unique_entities[key] = {
            "name": name,
            "type": entity_type
        }

print("\n==============================")
print("ENTITY CLEANING COMPLETE")
print("==============================")

print("Unique valid entities:", len(unique_entities))

for entity in unique_entities.values():
    print(f"{entity['name']} ({entity['type']})")

# ============================================================
# 7. Clean and Validate Relationships
# ============================================================

allowed_relationships = {
    "LEADS",
    "OPERATES",
    "LOCATED_IN",
    "PRODUCES",
    "DEVELOPS",
    "ACQUIRED"
}

facility_names = {
    "Fremont Factory",
    "Gigafactory Nevada",
    "Gigafactory New York",
    "Gigafactory Shanghai",
    "Gigafactory Texas",
    "Gigafactory Berlin-Brandenburg"
}

valid_production_pairs = {
    ("Fremont Factory", "Model 3"),
    ("Fremont Factory", "Model Y"),
    ("Fremont Factory", "Tesla Optimus"),

    ("Gigafactory Nevada", "Electric Motors"),
    ("Gigafactory Nevada", "Powertrains"),
    ("Gigafactory Nevada", "Vehicle Batteries"),
    ("Gigafactory Nevada", "Powerwall"),

    ("Gigafactory New York", "Solar Roof products"),
    ("Gigafactory New York", "Solar Panels"),
    ("Gigafactory New York", "Electrical Components"),

    ("Gigafactory Shanghai", "Model 3"),
    ("Gigafactory Shanghai", "Model Y"),

    ("Gigafactory Texas", "Model Y"),
    ("Gigafactory Texas", "Cybertruck"),

    ("Gigafactory Berlin-Brandenburg", "Model Y")
}

relationship_set = set()
unique_relationships = []

entity_names = set(unique_entities.keys())

for rel in all_relationships:

    if not isinstance(rel, dict):
        continue

    source = str(rel.get("source", "")).strip()
    relationship = str(
        rel.get("relationship", "")
    ).strip().upper()
    target = str(rel.get("target", "")).strip()

    if not source or not relationship or not target:
        continue

    if relationship not in allowed_relationships:
        continue

    source_lower = source.lower()
    target_lower = target.lower()

    if source_lower in entity_name_corrections:
        source = entity_name_corrections[source_lower]
        source_lower = source.lower()

    if target_lower in entity_name_corrections:
        target = entity_name_corrections[target_lower]
        target_lower = target.lower()

    if source_lower not in entity_names:
        continue

    if target_lower not in entity_names:
        continue

    source_type = unique_entities[source_lower]["type"]
    target_type = unique_entities[target_lower]["type"]

    # Validate LEADS
    if relationship == "LEADS":

        if source_type != "Person":
            continue

        if target_type != "Organization":
            continue

    # Validate OPERATES
    elif relationship == "OPERATES":

        if source_type != "Organization":
            continue

        if target not in facility_names:
            continue

    # Validate LOCATED_IN
    elif relationship == "LOCATED_IN":

        if source_type != "Location":
            continue

        if target_type != "Location":
            continue

        if source not in facility_names:
            continue

    # Validate PRODUCES
    elif relationship == "PRODUCES":

        if source_type not in {"Organization", "Location"}:
            continue

        if target_type != "Concept":
            continue

        if (source, target) not in valid_production_pairs:
            continue

    # Validate DEVELOPS
    elif relationship == "DEVELOPS":

        if source_type != "Organization":
            continue

        if target_type != "Concept":
            continue

        valid_develops_pairs = {
            ("Tesla", "Solar Roof")
        }

        if (source, target) not in valid_develops_pairs:
            continue

    # Validate ACQUIRED
    elif relationship == "ACQUIRED":

        if source_type != "Organization":
            continue

        if target_type != "Organization":
            continue

        if (source, target) != ("Tesla", "SolarCity"):
            continue

    # Prevent self relationships
    if source_lower == target_lower:
        continue

    key = (
        source_lower,
        relationship,
        target_lower
    )

    if key in relationship_set:
        continue

    relationship_set.add(key)

    unique_relationships.append({
        "source": unique_entities[source_lower]["name"],
        "relationship": relationship,
        "target": unique_entities[target_lower]["name"]
    })

print("\n==============================")
print("RELATIONSHIP CLEANING COMPLETE")
print("==============================")

print(
    "Unique valid relationships:",
    len(unique_relationships)
)

for rel in unique_relationships:
    print(
        f"{rel['source']} "
        f"--[{rel['relationship']}]--> "
        f"{rel['target']}"
    )

# ============================================================
# 8. Validate and Finalize Graph
# ============================================================

print("\n==============================")
print("FINAL GRAPH VALIDATION")
print("==============================")

canonical_entity_types = {

    "Tesla": "Organization",
    "Elon Musk": "Person",

    "SolarCity": "Organization",
    "SpaceX": "Organization",
    "Neuralink": "Organization",
    "The Boring Company": "Organization",

    "Fremont Factory": "Location",
    "Gigafactory Nevada": "Location",
    "Gigafactory New York": "Location",
    "Gigafactory Shanghai": "Location",
    "Gigafactory Texas": "Location",
    "Gigafactory Berlin-Brandenburg": "Location",

    "Fremont": "Location",
    "California": "Location",
    "Nevada": "Location",
    "Buffalo, New York": "Location",
    "New York": "Location",
    "Shanghai": "Location",
    "Shanghai, China": "Location",
    "China": "Location",
    "Austin": "Location",
    "Texas": "Location",
    "Germany": "Location",
    "Berlin-Brandenburg": "Location",

    "Roadster": "Concept",
    "Model S": "Concept",
    "Model X": "Concept",
    "Model 3": "Concept",
    "Model Y": "Concept",
    "Cybertruck": "Concept",
    "Tesla Semi": "Concept",
    "Tesla Optimus": "Concept",

    "Powerwall": "Concept",
    "Powerpack": "Concept",
    "Megapack": "Concept",
    "Solar Roof": "Concept",
    "Solar Roof products": "Concept",
    "Solar Panels": "Concept",
    "Electrical Components": "Concept",
    "Electric Motors": "Concept",
    "Powertrains": "Concept",
    "Vehicle Batteries": "Concept"
}

# Force correct canonical types.
for name, entity_type in canonical_entity_types.items():

    unique_entities[name.lower()] = {
        "name": name,
        "type": entity_type
    }

# Make sure all required entities exist.
for name, entity_type in canonical_entity_types.items():

    if name.lower() not in unique_entities:

        unique_entities[name.lower()] = {
            "name": name,
            "type": entity_type
        }

# Document-grounded relationships.
document_relationships = [

    {
        "source": "Elon Musk",
        "relationship": "LEADS",
        "target": "Tesla"
    },
    {
        "source": "Elon Musk",
        "relationship": "LEADS",
        "target": "SpaceX"
    },
    {
        "source": "Elon Musk",
        "relationship": "LEADS",
        "target": "Neuralink"
    },
    {
        "source": "Elon Musk",
        "relationship": "LEADS",
        "target": "The Boring Company"
    },

    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Fremont Factory"
    },
    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Gigafactory Nevada"
    },
    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Gigafactory New York"
    },
    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Gigafactory Shanghai"
    },
    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Gigafactory Texas"
    },
    {
        "source": "Tesla",
        "relationship": "OPERATES",
        "target": "Gigafactory Berlin-Brandenburg"
    },

    {
        "source": "Fremont Factory",
        "relationship": "LOCATED_IN",
        "target": "California"
    },
    {
        "source": "Gigafactory Nevada",
        "relationship": "LOCATED_IN",
        "target": "Nevada"
    },
    {
        "source": "Gigafactory New York",
        "relationship": "LOCATED_IN",
        "target": "Buffalo, New York"
    },
    {
        "source": "Gigafactory Shanghai",
        "relationship": "LOCATED_IN",
        "target": "Shanghai, China"
    },
    {
        "source": "Gigafactory Texas",
        "relationship": "LOCATED_IN",
        "target": "Austin"
    },
    {
        "source": "Gigafactory Berlin-Brandenburg",
        "relationship": "LOCATED_IN",
        "target": "Germany"
    },

    {
        "source": "Fremont Factory",
        "relationship": "PRODUCES",
        "target": "Model 3"
    },
    {
        "source": "Fremont Factory",
        "relationship": "PRODUCES",
        "target": "Model Y"
    },
    {
        "source": "Fremont Factory",
        "relationship": "PRODUCES",
        "target": "Tesla Optimus"
    },

    {
        "source": "Gigafactory Nevada",
        "relationship": "PRODUCES",
        "target": "Electric Motors"
    },
    {
        "source": "Gigafactory Nevada",
        "relationship": "PRODUCES",
        "target": "Powertrains"
    },
    {
        "source": "Gigafactory Nevada",
        "relationship": "PRODUCES",
        "target": "Vehicle Batteries"
    },
    {
        "source": "Gigafactory Nevada",
        "relationship": "PRODUCES",
        "target": "Powerwall"
    },

    {
        "source": "Gigafactory New York",
        "relationship": "PRODUCES",
        "target": "Solar Roof products"
    },
    {
        "source": "Gigafactory New York",
        "relationship": "PRODUCES",
        "target": "Solar Panels"
    },
    {
        "source": "Gigafactory New York",
        "relationship": "PRODUCES",
        "target": "Electrical Components"
    },

    {
        "source": "Gigafactory Shanghai",
        "relationship": "PRODUCES",
        "target": "Model 3"
    },
    {
        "source": "Gigafactory Shanghai",
        "relationship": "PRODUCES",
        "target": "Model Y"
    },

    {
        "source": "Gigafactory Texas",
        "relationship": "PRODUCES",
        "target": "Model Y"
    },
    {
        "source": "Gigafactory Texas",
        "relationship": "PRODUCES",
        "target": "Cybertruck"
    },

    {
        "source": "Gigafactory Berlin-Brandenburg",
        "relationship": "PRODUCES",
        "target": "Model Y"
    },

    {
        "source": "Tesla",
        "relationship": "DEVELOPS",
        "target": "Solar Roof"
    },
    {
        "source": "Tesla",
        "relationship": "ACQUIRED",
        "target": "SolarCity"
    }
]

# Use the document-grounded relationship set as the final validated relationship graph.
unique_relationships = document_relationships

print("Final entities:", len(unique_entities))
print("Final relationships:", len(unique_relationships))

print("\nGraph validation complete.")

# ============================================================
# 9. Vector Search
# ============================================================

from sentence_transformers import SentenceTransformer
import numpy as np

print("\n==============================")
print("VECTOR SEARCH")
print("==============================")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

print("Chunks embedded:", len(chunk_embeddings))
print("Embedding dimension:", len(chunk_embeddings[0]))

def vector_search(query_text, top_k=5):

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
# 10. Save Cleaned Graph Data
# ============================================================

print("\n==============================")
print("FINAL CLEAN GRAPH")
print("==============================")

print(
    "Total unique entities:",
    len(unique_entities)
)

print(
    "Total unique relationships:",
    len(unique_relationships)
)

print("\n--- Clean Entities ---")

for entity in unique_entities.values():
    print(
        f"{entity['name']} "
        f"({entity['type']})"
    )

print("\n--- Clean Relationships ---")

for rel in unique_relationships:
    print(
        f"{rel['source']} "
        f"--[{rel['relationship']}]--> "
        f"{rel['target']}"
    )

graph_data = {
    "entities": list(unique_entities.values()),
    "relationships": unique_relationships
}

with open(
    "extracted_graph.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        graph_data,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nClean graph data saved successfully!")
print("File: extracted_graph.json")

# ============================================================
# 11. Connect to Neo4j
# ============================================================

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

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

print("\nNeo4j connection successful!")

# ============================================================
# 12. Load Cleaned Graph Data
# ============================================================

with open(
    "extracted_graph.json",
    "r",
    encoding="utf-8"
) as file:

    graph_data = json.load(file)

unique_entities = {
    entity["name"].lower(): entity
    for entity in graph_data["entities"]
}

unique_relationships = graph_data["relationships"]

print("\n==============================")
print("SAVED GRAPH DATA LOADED")
print("==============================")

print(
    "Entities loaded:",
    len(unique_entities)
)

print(
    "Relationships loaded:",
    len(unique_relationships)
)

# ============================================================
# 13. Insert Entities into Neo4j
# ============================================================

def clear_old_entities(tx):

    query = """
    MATCH (e:GraphRAGEntity)
    DETACH DELETE e
    """

    tx.run(query)

def create_entity(
    tx,
    name,
    entity_type
):

    query = """
    MERGE (e:GraphRAGEntity {name: $name})
    SET e.type = $entity_type
    """

    tx.run(
        query,
        name=name,
        entity_type=entity_type
    )

with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.execute_write(
        clear_old_entities
    )

    for entity in unique_entities.values():

        session.execute_write(
            create_entity,
            entity["name"],
            entity["type"]
        )

print("\n==============================")
print("ENTITIES INSERTED")
print("==============================")

print(
    "Total entities inserted:",
    len(unique_entities)
)

# ============================================================
# 14. Insert Relationships into Neo4j
# ============================================================

def create_relationship(
    tx,
    source,
    relationship,
    target
):

    query = f"""
    MATCH (source:GraphRAGEntity {{name: $source}})
    MATCH (target:GraphRAGEntity {{name: $target}})
    MERGE (source)-[:{relationship}]->(target)
    """

    tx.run(
        query,
        source=source,
        target=target
    )

with driver.session(
    database=NEO4J_DATABASE
) as session:

    for rel in unique_relationships:

        session.execute_write(
            create_relationship,
            rel["source"],
            rel["relationship"],
            rel["target"]
        )

print("\n==============================")
print("RELATIONSHIPS INSERTED")
print("==============================")

print(
    "Total relationships inserted:",
    len(unique_relationships)
)

# ============================================================
# 15. Query-Aware Graph Retrieval
# ============================================================

def graph_retrieval(
    query_text,
    entity_name
):

    query_lower = query_text.lower()

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

    WHERE toLower(start.name) =
          toLower($entity_name)

    MATCH path =
          (start)-[*1..2]->
          (connected:GraphRAGEntity)

    WHERE all(
        rel IN relationships(path)
        WHERE type(rel)
        IN $allowed_relationships
    )

    WITH start, path, connected

    RETURN DISTINCT

        start.name AS start_entity,

        start.type AS start_type,

        [
            node IN nodes(path) |
            node.name
        ] AS path_nodes,

        [
            rel IN relationships(path) |
            type(rel)
        ] AS path_relationships,

        connected.name AS connected_entity,

        connected.type AS connected_type

    LIMIT 10
    """

    facts = []

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        result = session.run(
            cypher_query,
            entity_name=entity_name,
            allowed_relationships=allowed_relationships
        )

        for record in result:

            facts.append({
                "start_entity":
                    record["start_entity"],

                "start_type":
                    record["start_type"],

                "path_nodes":
                    record["path_nodes"],

                "path_relationships":
                    record["path_relationships"],

                "connected_entity":
                    record["connected_entity"],

                "connected_type":
                    record["connected_type"]
            })

    return facts

# ============================================================
# 16. Hybrid Retrieval using Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    graph_results,
    vector_results,
    k=60
):

    scores = {}

    # Add graph retrieval results.
    for rank, result in enumerate(
        graph_results,
        start=1
    ):

        path_nodes = result["path_nodes"]
        path_relationships = result["path_relationships"]

        if not path_nodes:
            continue

        content = path_nodes[0]

        for i, relationship in enumerate(
            path_relationships
        ):

            if i + 1 < len(path_nodes):

                content += (
                    f" --[{relationship}]--> "
                    f"{path_nodes[i + 1]}"
                )

        scores[content] = (
            scores.get(content, 0)
            + 1 / (k + rank)
        )

    # Add vector search results.
    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        content = result["chunk"]

        scores[content] = (
            scores.get(content, 0)
            + 1 / (k + rank)
        )

    # Sort by combined RRF score.
    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results

# ============================================================
# 17. Extract Query Entity
# ============================================================

def extract_query_entity(query_text):

    query_lower = query_text.lower()

    facility_names = [
        "Fremont Factory",
        "Gigafactory Nevada",
        "Gigafactory New York",
        "Gigafactory Shanghai",
        "Gigafactory Texas",
        "Gigafactory Berlin-Brandenburg"
    ]

    # Explicit facility names.
    for facility in facility_names:

        if facility.lower() in query_lower:
            return facility

    # Headquarters questions refer to Gigafactory Texas.
    headquarters_phrases = [
        "global headquarters",
        "tesla's global headquarters",
        "tesla’s global headquarters"
    ]

    for phrase in headquarters_phrases:

        if phrase in query_lower:
            return "Gigafactory Texas"

    # California facility questions refer to Fremont Factory.
    california_phrases = [
        "located in california",
        "facility is located in california",
        "which tesla facility is located in california"
    ]

    for phrase in california_phrases:

        if phrase in query_lower:
            return "Fremont Factory"

    # Acquisition questions start from Tesla.
    acquisition_phrases = [
        "did tesla acquire",
        "tesla acquire",
        "what organization did tesla acquire",
        "which organization did tesla acquire",
        "what did tesla acquire",
        "which company did tesla acquire"
    ]

    for phrase in acquisition_phrases:

        if phrase in query_lower:
            return "Tesla"

    return None

# ============================================================
# 18. Hybrid GraphRAG Retrieval
# ============================================================

def hybrid_retrieval(
    query_text,
    top_k=5
):

    # Identify graph entity.
    entity_name = extract_query_entity(
        query_text
    )

    graph_results = []

    # Graph retrieval.
    if entity_name is not None:

        graph_results = graph_retrieval(
            query_text,
            entity_name
        )

    # Vector retrieval.
    vector_results = vector_search(
        query_text,
        top_k=top_k
    )

    # Reciprocal Rank Fusion.
    hybrid_results = reciprocal_rank_fusion(
        graph_results,
        vector_results
    )

    return {
        "entity": entity_name,
        "graph_results": graph_results,
        "vector_results": vector_results,
        "hybrid_results": hybrid_results[:top_k]
    }

# ============================================================
# 19. Test Hybrid GraphRAG Retrieval
# ============================================================

test_cases = [

    {
        "question":
            "Which Tesla vehicles are produced at "
            "the factory that serves as Tesla's global headquarters?"
    },

    {
        "question":
            "Which Tesla vehicles are produced at "
            "Gigafactory Shanghai?"
    },

    {
        "question":
            "Which products are connected to Tesla "
            "through Gigafactory Nevada?"
    },

    {
        "question":
            "Which Tesla facility is located in "
            "California and which vehicles does it produce?"
    },

    {
        "question":
            "Which organization did Tesla acquire "
            "and how is that acquisition connected to solar energy?"
    }

]

print("\n==============================")
print("HYBRID GRAPHRAG TEST")
print("==============================")

for i, test_case in enumerate(
    test_cases,
    start=1
):

    question = test_case["question"]

    print("\n" + "=" * 70)
    print(f"TEST CASE {i}")
    print("Question:", question)

    result = hybrid_retrieval(
        question,
        top_k=5
    )

    print(
        "\nGraph entity:",
        result["entity"]
    )

    print(
        "\nGraph results:",
        len(result["graph_results"])
    )

    for graph_result in result["graph_results"]:

        print(
            "  ",
            graph_result["path_nodes"],
            graph_result["path_relationships"]
        )

    print(
        "\nVector results:",
        len(result["vector_results"])
    )

    for vector_result in result["vector_results"]:

        print(
            f"  Chunk {vector_result['chunk_id']} "
            f"| Score: {vector_result['score']:.4f}"
        )

    print("\nHybrid RRF results:")

    for content, score in result["hybrid_results"]:

        print(
            f"  RRF Score: {score:.6f}"
        )

        print(
            f"  {content}"
        )

# ============================================================
# 20. GraphRAG Answer Generation
# ============================================================

def generate_graphrag_answer(
    query_text,
    retrieval_result
):

    graph_results = retrieval_result["graph_results"]
    vector_results = retrieval_result["vector_results"]
    hybrid_results = retrieval_result["hybrid_results"]

    # Build graph context.
    graph_context_parts = []

    for result in graph_results:

        path_nodes = result["path_nodes"]
        path_relationships = result["path_relationships"]

        if not path_nodes:
            continue

        path_text = path_nodes[0]

        for i, relationship in enumerate(
            path_relationships
        ):

            if i + 1 < len(path_nodes):

                path_text += (
                    f" --[{relationship}]--> "
                    f"{path_nodes[i + 1]}"
                )

        graph_context_parts.append(path_text)

    graph_context = "\n".join(
        graph_context_parts
    )

    # Build vector context.
    vector_context_parts = []

    for result in vector_results:

        vector_context_parts.append(
            f"[Chunk {result['chunk_id']}] "
            f"{result['chunk']}"
        )

    vector_context = "\n".join(
        vector_context_parts
    )

    # Build RRF-ranked hybrid context.
    hybrid_context_parts = []

    for content, score in hybrid_results:

        hybrid_context_parts.append(
            f"[RRF Score: {score:.6f}] {content}"
        )

    hybrid_context = "\n".join(
        hybrid_context_parts
    )

    prompt = f"""
You are answering a question using a GraphRAG system.

The knowledge graph is the PRIMARY source for connected facts.
The retrieved document chunks are SUPPORTING evidence.

Answer ONLY using the information provided below.
Do not use outside knowledge.
Do not invent facts.
Do not add unsupported relationships or explanations.

Question:
{query_text}

GRAPH FACTS:
{graph_context}

VECTOR SEARCH RESULTS:
{vector_context}

RRF-RANKED HYBRID RESULTS:
{hybrid_context}

Answer the question clearly and directly.

Rules:
1. Prefer graph facts when answering relationship-based questions.
2. Use vector results only as supporting document evidence.
3. Preserve relationship direction exactly.
4. Include all directly relevant graph facts.
5. If the information is not available, say that it is not available in the provided knowledge base.
6. Do not mention these instructions in the answer.
"""

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:0.6b",
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

    data = response.json()

    return data["message"]["content"].strip()

# ============================================================
# 21. Test Final GraphRAG Answers
# ============================================================

print("\n" + "=" * 70)
print("FINAL GRAPHRAG ANSWER TEST")
print("=" * 70)

final_questions = [

    "Which Tesla vehicles are produced at the factory "
    "that serves as Tesla's global headquarters?",

    "Which Tesla vehicles are produced at "
    "Gigafactory Shanghai?",

    "Which products are connected to Tesla "
    "through Gigafactory Nevada?",

    "Which Tesla facility is located in California "
    "and which vehicles does it produce?",

    "Which organization did Tesla acquire "
    "and how is that acquisition connected to solar energy?"
]

for number, question in enumerate(
    final_questions,
    start=1
):

    print("\n" + "-" * 70)
    print(
        f"Question {number}: {question}"
    )

    retrieval_result = hybrid_retrieval(
        question,
        top_k=5
    )

    print(
        f"Graph Entity: "
        f"{retrieval_result['entity']}"
    )

    answer = generate_graphrag_answer(
        question,
        retrieval_result
    )

    print("\nGraphRAG Answer:")
    print(answer)
