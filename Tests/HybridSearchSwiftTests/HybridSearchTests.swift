import Foundation
import Testing
@testable import HybridSearchSwift

@TantivyDocument
struct TestDoc: Sendable {
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

private enum TestData {
    static let docs: [TestDoc] = [
        TestDoc(
            id: "swift-1",
            title: "Swift Concurrency",
            body: "Async await and actors in Swift.",
            isPublished: true
        ),
        TestDoc(
            id: "rust-1",
            title: "Rust FFI",
            body: "Calling Rust from Swift using UniFFI.",
            isPublished: true
        ),
        TestDoc(
            id: "vector-1",
            title: "Vector Search with HNSW",
            body: "Approximate nearest neighbor search using HNSW graphs.",
            isPublished: false
        ),
        TestDoc(
            id: "tantivy-1",
            title: "Full-text Search with Tantivy",
            body: "Indexing and BM25 ranking with Tantivy.",
            isPublished: true
        ),
    ]

    static let embeddings: [[Float]] = [
        TestEmbeddings.docSwiftConcurrency,
        TestEmbeddings.docRustFfi,
        TestEmbeddings.docVectorHnsw,
        TestEmbeddings.docTantivy,
    ]
}

private func makeIndex() throws -> HybridIndex<TestDoc> {
    let baseURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)

    let config = HybridIndexConfig(
        embeddingDimension: 128,
        hnswMaxConnections: 16,
        hnswMaxElements: 1000,
        hnswMaxLayers: 16,
        hnswEfConstruction: 200,
        distanceType: .cosine
    )

    return try HybridIndex(path: baseURL.path, config: config)
}

private func seedIndex(_ index: HybridIndex<TestDoc>) async throws -> [UInt64] {
    let pairs = zip(TestData.docs, TestData.embeddings).map { ($0.0, $0.1) }
    let docIds = try await index.add(docs: pairs)
    try await index.commit()
    return docIds
}

@Suite("HybridSearchSwift Tests")
struct HybridSearchSwiftTests {
    @Test("Index and fetch by ID")
    func testIndexAndFetch() async throws {
        let index = try makeIndex()
        let docIds = try await seedIndex(index)

        let fetched = try await index.get(idField: .id, idValue: .text("swift-1"))
        #expect(fetched?.title == "Swift Concurrency")

        let byDocId = try await index.get(docId: docIds[0])
        #expect(byDocId?.id == "swift-1")
    }

    @Test("Text search returns Swift result")
    func testTextSearch() async throws {
        let index = try makeIndex()
        _ = try await seedIndex(index)

        let query = HybridTextQuery<TestDoc>(
            query: "swift actors",
            defaultFields: [.title, .body]
        )
        let results = try await index.searchText(query: query, limit: 3)
        #expect(results.first?.document.id == "swift-1")
    }

    @Test("Vector search applies filters")
    func testVectorSearchFilter() async throws {
        let index = try makeIndex()
        _ = try await seedIndex(index)

        let filter = TantivyQuery.term(
            TantivyQueryTerm(name: "isPublished", value: .bool(true))
        )
        let results = try await index.searchVector(
            embedding: TestEmbeddings.queryVector,
            filter: filter,
            limit: 3,
            efSearch: 50
        )

        #expect(!results.isEmpty)
        #expect(results.allSatisfy { $0.document.isPublished })
        #expect(results.first?.document.id != "vector-1")
    }

    @Test("Hybrid search uses RRF")
    func testHybridSearch() async throws {
        let index = try makeIndex()
        _ = try await seedIndex(index)

        let query = HybridTextQuery<TestDoc>(
            query: "swift concurrency actors",
            defaultFields: [.title, .body]
        )
        let results = try await index.searchHybrid(
            query: query,
            embedding: TestEmbeddings.querySwift,
            limit: 3,
            efSearch: 50
        )

        #expect(results.first?.document.id == "swift-1")
    }

    @Test("Delete removes documents")
    func testDelete() async throws {
        let index = try makeIndex()
        _ = try await seedIndex(index)

        try await index.delete(idField: .id, idValue: .text("rust-1"))
        let fetched = try await index.get(idField: .id, idValue: .text("rust-1"))
        #expect(fetched == nil)

        let query = HybridTextQuery<TestDoc>(
            query: "Rust",
            defaultFields: [.title, .body]
        )
        let results = try await index.searchText(query: query, limit: 5)
        #expect(results.allSatisfy { $0.document.id != "rust-1" })
    }
}
