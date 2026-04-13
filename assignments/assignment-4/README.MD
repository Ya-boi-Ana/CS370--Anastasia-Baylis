## Anastasia Baylis
## CS370-102
## Assignment 4
## 4/12/2026



# 1. Deep Document understanding vs. naive chunking
Deep document understanding is an AI-powered process that goes beyond the traditional OCR that allows the extraction of data from different types of structures, even unstructured PDFS, images, and scans. Naive chunking is the standard simplistic method of breaking documents into large chunks, and then into smaller chunks.

Naive chunking is optimized for token budget, not semantic boundaries that breaks:
Boundary integrity, Attribution integrity, Reference integrity. Layout aware parsing improves retrieval fidelity because the unit being indexed is closer to the unit a human would cite. In RAGFlow, spreadsheet parsing preserves the layout and table structure and will later release the added contextual information around images and tables, with a parent slicing strategy that is aimed at preserving the local meaning of the information, for retrieval.

Deep parsing changes index from “text with embeddings” into a more in depth object with: text span, structural role, source coordinate or page locality, parent-child relationships, and lastly metadata and generated enrichments. RAGflow’s pipeline separates the parser, transformer, chunker, indexer, and docs note that parsing includes building both embedding and full text indexed on the chunks produced. The separation is a signal that the index is retrieval optimized, rather than a dump of source text.

The downside is that deep document understanding is more expensive and takes longer, while naive chunking is cheaper it cannot do as much as deep document understanding.

# 2. Chunking Strategy: template vs semantic
RAGflow supports multiple built-in chunking templates, instead of 1 universal template. Template chunking uses a document type assumption, so it can differentiate between a table meaning it has columns and rows, presentations meaning there's slides, and a paper will just be predictable.

The strengths of template chunking is: low variance, easy to tune, and stable indexing while some weaknesses are: doesn’t work well when the template isn’t followed 100%, needs a document classification, and it can miss topic transitions.

Semantic segmentation chunking tries to create boundaries when theres a shift, some strengths of this are: adapts to a loosely structured text, handles changes in topics better, and performs better on long notes or transcripts. Some of its weaknesses are: its expensive, boundary placements are dependent on the model, and it can ignore structures like table rows.

What causes failures? Mostly a document with too much structure like a financial document. Semantic will be the most likely to fail, as its less dependent on a layout and just wants to create boundaries, so if it is given a financial excel document that depends on the column and row labels this can cause the semantic to fail due to it possibly merging columns and rows that are distinct simply because they are both about similar topics. Template chunking is less likely to fail in this situation due to its dependence on a structure. When it comes to a document with less structure, like screen shots of a group chat, template chunking fails more often, as it is so dependent on a structure, while semantic will excel because it can create boundaries based on the topic, not on types. Overall it is best to use semantic with informative information, like a textbook or a text chat, while template chunking is best for very template heavy information like a spreadsheet.
 
# 3. 

# 4. 

# 5. 

# 6. 

# 7. 

# 8. 

# 9. 

# 10.