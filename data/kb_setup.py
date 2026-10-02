"""
CRRA Lab C1 — Policy Knowledge Base Setup

Loads the BizOps procurement policy articles into a ChromaDB collection so the
Analysis Agent (Lab C3) can cite policy when it makes a recommendation.

Run from the project root:
    python data/kb_setup.py
"""

import chromadb
from pathlib import Path

KB_DIR = Path(__file__).parent / "kb"
# Saved to disk so the Lab C3 agent can query it after this script exits
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "crra_policy"


def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a policy article at '## ' headings.

    Each policy article covers several distinct rules. Embedding a whole file as
    one chunk buries the specific rule you need inside a wall of unrelated text,
    and the search returns weak matches for everything. Splitting at section
    headings keeps each rule independently searchable.
    """
    chunks = []
    current_heading = "Overview"
    current_body: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                body = "\n".join(current_body).strip()
                if body:
                    chunks.append({"heading": current_heading, "text": body})
            current_heading = line[3:].strip()
            current_body = []
        elif line.startswith("# "):
            continue  # document title, not a section
        else:
            current_body.append(line)

    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            chunks.append({"heading": current_heading, "text": body})

    return [
        {
            "id": f"{filename}::{i}",
            "document": f"{c['heading']}\n{c['text']}",
            "metadata": {"source": filename, "heading": c["heading"]},
        }
        for i, c in enumerate(chunks)
    ]


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Start clean so re-running the script does not stack duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine distance runs 0..1 for these embeddings, so (1 - distance) is a
    # true 0-100% similarity. Chroma's default (squared L2) runs 0..4 and would
    # give negative "confidence" for weak matches.
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR} — check the folder path.")

    all_ids, all_docs, all_meta = [], [], []
    print("Loading policy articles\n" + "=" * 55)

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_article(text, md_file.name)
        print(f"  {md_file.name:<32} {len(chunks)} chunks")
        for c in chunks:
            all_ids.append(c["id"])
            all_docs.append(c["document"])
            all_meta.append(c["metadata"])

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_meta)
    print("=" * 55)
    print(f"  TOTAL: {len(all_ids)} chunks from {len(md_files)} articles\n")

    # ---- Verify retrieval actually works, with realistic BizOps questions ----
    test_queries = [
        "who approves a contract worth 60 lakh rupees",
        "contract auto renews next month and we missed the notice deadline",
        "two monitoring tools with low licence usage",
        "vendor wants a 20 percent price increase at renewal",
        "we no longer need this tool at all, how do we exit",
        # Break-it tests
        "what is the travel expense limit",  # not covered by the policy
        "can I terminate a 90 lakh contract early",  # spans two articles
    ]

    print("Testing retrieval\n" + "=" * 55)
    for q in test_queries:
        res = collection.query(query_texts=[q], n_results=1)
        source = res["metadatas"][0][0]["source"]
        heading = res["metadatas"][0][0]["heading"]
        confidence = 1 - res["distances"][0][0]
        print(f'  "{q[:46]}..."')
        print(f"     -> {source}  §{heading}   confidence {confidence:.0%}\n")

    print("=" * 55)
    print("KB ready. Lab C3's Analysis Agent will query this collection.")


if __name__ == "__main__":
    main()
