# HybridSearchSwift

HybridSearchSwift combines **Tantivy full-text search** and **HNSW vector search** in Swift and merges results using **Reciprocal Rank Fusion (RRF)**. It is built on top of `tantivy.swift` and `HNSW.swift`, and keeps both indices on disk.

This implementation uses a **simple query-string API + optional filter DSL** (Option B) so you can start fast while still applying structured filters.

## Features

- **Hybrid search** (BM25 + vector) with RRF
- **Text-only** (BM25) and **vector-only** search
- **Structured filters** using Tantivy query DSL
- **Index / delete / fetch by ID**
- **Disk persistence** for both indices
- **Swift-native schema definition** with `@TantivyDocument`

## Installation

```swift
.package(path: "../HybridSearch.swift")
```

Depends on:

- [tantivy.swift](https://github.com/botisan-ai/tantivy.swift)
- [HNSW.swift](https://github.com/botisan-ai/HNSW.swift)

## Quick Start

### 1. Define a document

```swift
import HybridSearchSwift

@TantivyDocument
struct Article: Sendable {
    @IDField var id: String
    @TextField var title: String
    @TextField var body: String
    @BoolField var isPublished: Bool

    init(id: String, title: String, body: String, isPublished: Bool) {
        self.id = id
        self.title = title
        self.body = body
        self.isPublished = isPublished
    }
}
```

> **Requirement:** each document must have **at least one `@IDField`**. If multiple ID fields exist, pass `primaryIdField:` when creating/loading the index.

### 2. Create an index

```swift
let index = try HybridIndex<Article>(
    path: "./articles-index",
    config: HybridIndexConfig(
        embeddingDimension: 384,
        distanceType: .cosine
    )
)
```

### 3. Index documents

```swift
let article = Article(
    id: "a1",
    title: "Hybrid Search",
    body: "Combining BM25 and vector search with RRF",
    isPublished: true
)

let embedding: [Float] = embed(article.title + " " + article.body)

try await index.add(doc: article, embedding: embedding)
try await index.commit()
```

### 4. Search (text, vector, hybrid)

#### Text-only (BM25)

```swift
let query = HybridTextQuery<Article>(
    query: "hybrid search",
    defaultFields: [.title, .body]
)

let results = try await index.searchText(query: query, limit: 10)
```

#### Vector-only

```swift
let results = try await index.searchVector(
    embedding: queryEmbedding,
    limit: 10,
    efSearch: 100
)
```

#### Hybrid (RRF)

```swift
let results = try await index.searchHybrid(
    query: query,
    embedding: queryEmbedding,
    limit: 10,
    efSearch: 100
)
```

## Filters (DSL)

Filters are expressed using Tantivy’s DSL and can be applied to **text**, **vector**, or **hybrid** searches.

```swift
let filter = TantivyQuery.term(
    TantivyQueryTerm(name: "isPublished", value: .bool(true))
)

let results = try await index.searchVector(
    embedding: queryEmbedding,
    filter: filter,
    limit: 10
)
```

For hybrid search, filters are applied to **both** the BM25 and vector sides before RRF fusion.

## Fetch & Delete

```swift
// Fetch by external ID
let doc = try await index.get(idField: .id, idValue: .text("a1"))

// Fetch by internal docId
let doc2 = try await index.get(docId: 42)

// Delete by external ID
try await index.delete(idField: .id, idValue: .text("a1"))
```

## RRF Defaults

Hybrid ranking uses:

```
score = textWeight / (k + rank) + vectorWeight / (k + rank)
```

Defaults:

- `k = 60`
- `textWeight = 1`
- `vectorWeight = 1`

You can override these in `searchHybrid(...)`.

## Persistence

The index stores:

```
index-directory/
├── tantivy/
├── hnsw.data
├── hnsw.graph
└── hybrid.meta.json
```

Load with:

```swift
let index = try HybridIndex<Article>(loadFrom: "./articles-index")
```

## Notes

- For cosine similarity, **normalize embeddings** if your model doesn’t already.
- `efSearch` should generally be ≥ `k`.
- Vector filtering is done by intersecting HNSW candidate IDs with Tantivy DSL filters.

## Development

```bash
cd HybridSearch.swift
swift test
```
