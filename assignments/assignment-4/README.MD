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
 
# 3. Hybrid Retrieval architecture 
RAGflow uses 3 retrieval steps together:
* Lexical search ~ exact language
* Vector search ~ uses embeddings
* Re-ranking ~ sorts the best matches higher
This works better than using a singular method, since each method is good at different things.

Hybrid retrieval is better because: lexical search is best when the users question has the same words as the document, while vector search is best when the question is on the same topic but not the same exact wording in the document. When these two types are used together you get better recall and better precision.

Lexical only search can fail if a user uses different wording to mean the same thing on the document like: “angle foodcake” and “angel food cake” if the user were to accidentally use a typo. While vector-only search will fail when the exact wording is important, like if you were searching for a “strawberry cheesecake recipe” and it gave you a generalized cheesecake recipe instead of the intended. Hybrid may only fail if the document was badly parsed, say if a PDF were to be broken or rows and columns get mixed up. Hybrid is overall the best due to its ability to use both lexical and vector search, and may only be truly broken if it was given a bad document.

# 4. Multi-stage retrieval pipeline
RAGflow does retrieval in multiple steps instead of 1 big step:
Find many possible documents (candidate generation)
Sort them (re-ranking)
Improve the query if needed (query refinement)

Single pass ANN is fast, however, it tries to find relevant documents and rank them correctly all at once, which makes it less accurate compared to Multi-stage. Multi-stage improves the quality of the retrieval but it takes more time, so oftentimes the systems use fewer steps to make it faster. One problem with multi-stage is that the errors can spread throughout the entire pipeline and the later steps cannot fix it at all, which can cause ban querying, re-ranking can choose the wrong document, and extra steps can make it much more complex.

# 5. Indexing strategy and storage backends
RAGflow uses retrieval optimized indexes instead of a normal database which makes searches faster. Elasticsearch is a hybrid store, and is best when you need both a keyword search and a vector search. Vector-native databases are best when the system relies on semantic similarity search. The strength of this is a very fast semantic search, however a weakness for exact keyword matching. Graph augmented store is best when the relationship between the entities matter, like in financial data. Graph-store’s strength is supporting multi-step reasoning, but it is much harder to build and maintain.

# 6. Query understanding and reformation
RAGflow improves retrieval by rewriting or expanding on the users query before searching. Static query retrieval uses the user’s original query to search directly and is best when the question is clear, the wording matches the documents, and the conversation is 1 step. This is fast and simple however, it fails if the query is too vague or incomplete. Iterative query refinement improves the query step by step using context or an agent and is best when the questions depend on a prior conversation or if the question has missing details, the strength of this is a more accurate retrieval, but is slower and requires more computation. This improves RAGflow because it fills in missing information and clarifies the users meaning, however it also increases the response time and is best used only when needed.

# 7. Knowledge representation layer
RAGflow stores knowledge in different formats so the system can retrieve and use the information better, the main types are:
* Dense vector space
* Relational schema
* Knowledge graph
Dense vector space is best for semantic searches, paraphrased question and general document retrieval and its strength is finding similar meaning with different wording but its weakness is: it has a hard time explaining why something was retrieved. Relational schema stores knowledge in structured rows and columns and is best for exact facts, numbers, filters, business data and its strength is that its precise and reliable, but its weakness is that its not good for anything unstructured. Knowledge Graph stores entities and the relationships between them and is best for multi-step reasoning, relationship questions, and complex topics with its strengths being that it supports connection based reasoning, however its weakness is that its harder to build and hard to maintain. Overall, its best to combine all of these for the best results depending on the users questions and topics.

# 8. Data ingestion pipeline architecture
RAGflow uses an ingestion pipeline to turn raw documents into searchable knowledge. A good ingestion pipeline should be able to handle different file types, handle indexes efficiently, and balance speed with accuracy. First data must be normalized across all sources so that there is a standard format instead of 8 different ones causing retrieval to be consistent however, it requires extra preprocessing time, second it must update the indexes incrementally instead of reprocessing, it simply changes the indexes of things it updated in the first step which allows for faster updates and lower computation costs but creates more complex pipeline logic, lastly the system must balance consistency (how fast the data shows up, and how accurate new results are) and throughput (faster processing of large datasets, slight delays). Overall most RAG systems are best off with higher throughput with smaller delays ensuring retrieval is fast, scalable, and reliable.

# 9. Memory design in RAG systems
Ragflow includes memory components that help the system remember prior interactions and information during long conversation, this improves multi-step reasoning as well as the personalization. 
There are 3 types of personalization:
* Vector memory
* Structured memory
* Eposiodic logs
Each supporting a different task

Vector memory is best for recalling related conversations, remembering user preferences, and finding similar past questions making this strong at being flexible and works with natural language, while it may get incorrect memories. Structured memory stores information in tables or knowledge graphs and is best for user profiles, task progress, saved facts, and the system state making it strong and reliable however its harder to update and design. Episodic logs store conversation history in time order and are best for tracking what happened earlier, debugging a conversation and explaining past actions. Its strength is that it keeps track of a full history, however its large and harder to search quickly. Overall each types of memory has a different role and its best for RAG systems to have all 3 for the best user experience.

# 10.End-to-end system decomposition
RAGflow includes several parts: ingestion, indexing, retrieval, reasoning, and serving. A good RAG system separates these into microservices so parts can scale or fail independently.

Stateless services to not store long term data, and are easy to scale they are used for API gateways, retrieval services, agent reasoning, and embedding service. Its strength is that it quickly adds more servers when demand increases. While stateful services store important system data and are used in vector database, keyword indexes, etc their strength is their persistent knowledge and conversation history. Different services scale differently depending on its workload so Ingestion service scales when there are many documents uploaded, embedding services scale when generating many vectors, retrieval services scale with user queries, agent services scale with conversation traffic. 

Services may fail, however due to the isolation of each service and their individual scaling it doesn’t affect other services, which makes the system much more reliable. Overall these microservices improve RAGflow by separating the stateless and stateful components, scaling services independently, and preventing failures from affecting the whole system, making it more flexible and reliable.
