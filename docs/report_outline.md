# Report Outline (≤10 pages)

1. **Problem Statement & Societal Context**
   - Digital trust risks from meme- and claim-based misinformation.
   - Need for correlation across modalities and sources.

2. **System Overview**
   - Claim-centric correlation engine.
   - Evidence-based outputs with traceability.

3. **Architecture**
   - Modules, data flow, and storage layers.
   - Qdrant as primary vector memory and retrieval substrate.

4. **Qdrant Design**
   - Collections: `claims`, `evidence_snippets`, `media_memes`.
   - Payload schema and metadata filtering.

5. **Multimodal Retrieval Strategy**
   - Meme OCR + CLIP image embeddings + text embeddings.
   - Query flows for meme and text.

6. **Long-Term Memory & Canonicalization**
   - Merge vs create logic.
   - Reinforcement, contradiction, decay.
   - SQLite audit events.

7. **Evidence-Based Responses**
   - Grounded outputs and trace panel.
   - Insufficient/conflicting evidence messaging.
   - Optional LLM deduction panel (Ollama) grounded in retrieved evidence.

8. **Societal Responsibility**
   - Limitations, bias, privacy, safe messaging.

9. **Demo Examples**
   - Sample queries with screenshots and trace logs.

10. **Future Work**
   - Improved stance classification, credibility scoring, and provenance.
