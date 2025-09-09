import marqo

def initialize_marqo_index():
    client = marqo.Client("http://localhost:8882")

    index_name = "rag_index"

    # Delete index if it already exists
    try:
        client.delete_index(index_name)
    except Exception:
        pass

    # Create unstructured index (no tensor_fields needed)
    client.create_index(
        index_name,
        type="unstructured"
    )

    # Add some sample documents
    documents = [
        {"_id": "1", "content": "Healthcare services include medical care, health screenings, and wellness programs."},
        {"_id": "2", "content": "Food assistance provides food banks, meal programs, and emergency groceries."},
        {"_id": "3", "content": "Transport services include subsidized rides, shuttle buses, and fuel assistance."},
        {"_id": "4", "content": "Housing support covers rental assistance, shelters, and affordable housing programs."}
    ]

    client.index(index_name).add_documents(documents)

    print("✅ rag_index created and documents added successfully!")
