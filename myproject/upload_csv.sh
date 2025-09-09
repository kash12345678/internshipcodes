#!/bin/bash
csv_file="data.csv"
index_name="community_assistance_index"
marqo_url="http://localhost:8882"

python3 - <<EOF
import csv, requests
csv_file = "$csv_file"
index_name = "$index_name"
marqo_url = "$marqo_url"
documents = []
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        documents.append({
            "content": row["content"],
            "topic": row["topic"],
            "references": row["references"].split(";")
        })
payload = {"documents": documents, "tensor_fields": []}
resp = requests.post(f"{marqo_url}/indexes/{index_name}/documents", json=payload)
print(resp.json())
EOF

