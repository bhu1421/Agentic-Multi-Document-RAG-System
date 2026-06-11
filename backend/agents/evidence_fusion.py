import hashlib


def evidence_fusion_node(state):
    """Intelligently merge evidence from local retrieval and web search.

    This is a **deterministic** agent (no LLM call) that:
    1. Tags each document with its origin (local vs. web)
    2. Deduplicates by exact content hash
    3. Removes near-duplicates (shorter doc fully contained in a longer one)
    4. Prioritizes local docs over web docs in the merged output

    Zero added latency from API calls.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    web_docs = state.get("web_docs", [])

    # --- Step 1: Tag provenance ---
    for doc in retrieved_docs:
        if "origin" not in doc.metadata:
            doc.metadata["origin"] = "local"

    for doc in web_docs:
        if "origin" not in doc.metadata:
            doc.metadata["origin"] = "web"

    # Local docs come first (higher trust), then web docs fill gaps
    all_docs = retrieved_docs + web_docs

    if not all_docs:
        print("[EvidenceFusion] No evidence to fuse")
        return {"fused_docs": []}

    # --- Step 2: Exact-hash deduplication ---
    seen_hashes = set()
    hash_deduped = []
    for doc in all_docs:
        content_hash = hashlib.md5(
            doc.page_content.strip().lower().encode()
        ).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            hash_deduped.append(doc)

    # --- Step 3: Near-duplicate removal ---
    # If a shorter doc's content is fully contained within a longer doc, drop it.
    # Only run for small batches to avoid O(n²) blowup on huge result sets.
    if len(hash_deduped) <= 50:
        final_docs = []
        for i, doc in enumerate(hash_deduped):
            is_subduplicate = False
            stripped = doc.page_content.strip()
            for j, other in enumerate(hash_deduped):
                if i != j and len(other.page_content) > len(stripped):
                    if stripped in other.page_content:
                        is_subduplicate = True
                        break
            if not is_subduplicate:
                final_docs.append(doc)
    else:
        final_docs = hash_deduped

    # --- Stats ---
    local_count = sum(1 for d in final_docs if d.metadata.get("origin") == "local")
    web_count = sum(1 for d in final_docs if d.metadata.get("origin") == "web")
    removed = len(all_docs) - len(final_docs)

    print(
        f"[EvidenceFusion] Input: {len(retrieved_docs)} local + {len(web_docs)} web "
        f"= {len(all_docs)} total"
    )
    print(
        f"[EvidenceFusion] Output: {len(final_docs)} unique docs "
        f"({removed} duplicates removed) | {local_count} local, {web_count} web"
    )

    return {"fused_docs": final_docs}
