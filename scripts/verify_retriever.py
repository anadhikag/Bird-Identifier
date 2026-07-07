"""
Verify species-aware retrieval works correctly after the architecture fix.

Run from the project root:
    venv\Scripts\python.exe scripts\verify_retriever.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.rag.retriever import Retriever

retriever = Retriever()

print("=" * 70)
print("TEST 1: Northern Cardinal — 'Where does it migrate?'")
print("=" * 70)
chunks = retriever.retrieve(
    query="Where does it migrate?",
    top_k=3,
    species="017.Cardinal",
)
if chunks:
    for c in chunks:
        print(f"  [{c.score:.4f}] {c.species} (chunk {c.chunk_index})")
        print(f"    {c.text[:120].replace(chr(10), ' ')}...")
    print("  PASS — Cardinal chunks returned")
else:
    print("  FAIL — No chunks returned for Cardinal!")

print()
print("=" * 70)
print("TEST 2: Blue Jay — 'What does it eat?'")
print("=" * 70)
chunks = retriever.retrieve(
    query="What does it eat?",
    top_k=3,
    species="073.Blue_Jay",
)
if chunks:
    for c in chunks:
        print(f"  [{c.score:.4f}] {c.species} (chunk {c.chunk_index})")
        print(f"    {c.text[:120].replace(chr(10), ' ')}...")
    print("  PASS — Blue Jay chunks returned")
else:
    print("  FAIL — No chunks returned for Blue Jay!")

print()
print("=" * 70)
print("TEST 3: Global search (no species) — 'bird migration patterns'")
print("=" * 70)
chunks = retriever.retrieve(
    query="bird migration patterns",
    top_k=5,
)
species_found = {c.species for c in chunks}
print(f"  Retrieved {len(chunks)} chunks from species: {species_found}")
if len(chunks) > 0:
    print("  PASS — Global search still works")
else:
    print("  FAIL — Global search broken!")

print()
print("=" * 70)
print("TEST 4: Cross-contamination check — Cardinal query should return ONLY Cardinal chunks")
print("=" * 70)
chunks = retriever.retrieve(
    query="what are the nesting habits and habitat of this bird?",
    top_k=5,
    species="017.Cardinal",
)
non_cardinal = [c for c in chunks if c.species != "017.Cardinal"]
if non_cardinal:
    print(f"  FAIL — Got chunks from wrong species: {[c.species for c in non_cardinal]}")
else:
    print(f"  PASS — All {len(chunks)} returned chunks belong to 017.Cardinal")
