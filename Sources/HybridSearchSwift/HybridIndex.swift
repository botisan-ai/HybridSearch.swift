import Foundation
import TantivySwift
import HnswSwift

public enum HybridSearchError: Error {
    case metadataMissing
    case metadataCorrupt
    case indexAlreadyExists
    case missingIdField
    case ambiguousIdField([String])
    case invalidPrimaryIdField(String)
    case dimensionMismatch(expected: UInt32, got: UInt32)
    case missingDocId
}

public struct HybridIndexConfig: Sendable {
    public var embeddingDimension: UInt32
    public var hnswMaxConnections: UInt32
    public var hnswMaxElements: UInt64
    public var hnswMaxLayers: UInt32
    public var hnswEfConstruction: UInt32
    public var distanceType: HnswDistanceType

    public init(
        embeddingDimension: UInt32 = 384,
        hnswMaxConnections: UInt32 = 16,
        hnswMaxElements: UInt64 = 100_000,
        hnswMaxLayers: UInt32 = 16,
        hnswEfConstruction: UInt32 = 200,
        distanceType: HnswDistanceType? = nil
    ) {
        self.embeddingDimension = embeddingDimension
        self.hnswMaxConnections = hnswMaxConnections
        self.hnswMaxElements = hnswMaxElements
        self.hnswMaxLayers = hnswMaxLayers
        self.hnswEfConstruction = hnswEfConstruction
        self.distanceType = distanceType ?? .cosine
    }

    var hnswConfig: HnswIndexConfig {
        HnswIndexConfig(
            maxConnections: hnswMaxConnections,
            maxElements: hnswMaxElements,
            maxLayers: hnswMaxLayers,
            efConstruction: hnswEfConstruction,
            dimension: embeddingDimension,
            distanceType: distanceType
        )
    }
}

public struct HybridSearchResult<Doc: TantivyDocument>: Sendable {
    public let docId: UInt64
    public let score: Float
    public let document: Doc

    public init(docId: UInt64, score: Float, document: Doc) {
        self.docId = docId
        self.score = score
        self.document = document
    }
}

private enum HybridConstants {
    static let docIdFieldName = "__doc_id"
    static let metadataFileName = "hybrid.meta.json"
    static let hnswBasename = "hnsw"
    static let defaultRrfK: Float = 60
}

private protocol HybridIDFieldMarker {}

extension IDField: HybridIDFieldMarker {}

private enum HybridSchemaInspector {
    static func idFieldNames<Doc: TantivyDocument>(for docType: Doc.Type) -> [String] {
        let instance = docType.schemaTemplate()
        let mirror = Mirror(reflecting: instance)
        return mirror.children.compactMap { child in
            guard let label = child.label else { return nil }
            let name = label.hasPrefix("_") ? String(label.dropFirst()) : label
            return child.value is HybridIDFieldMarker ? name : nil
        }
    }

    static func textFieldNames<Doc: TantivyDocument>(for docType: Doc.Type) -> [String] {
        let instance = docType.schemaTemplate()
        let mirror = Mirror(reflecting: instance)
        return mirror.children.compactMap { child in
            guard let label = child.label else { return nil }
            let name = label.hasPrefix("_") ? String(label.dropFirst()) : label
            return child.value is TantivyTextFieldMarker ? name : nil
        }
    }

    static func schemaFingerprint<Doc: TantivyDocument>(for docType: Doc.Type) -> String {
        let instance = docType.schemaTemplate()
        let mirror = Mirror(reflecting: instance)
        var parts: [String] = []

        for child in mirror.children {
            guard let label = child.label else { continue }
            let name = label.hasPrefix("_") ? String(label.dropFirst()) : label
            let wrapper = String(describing: Swift.type(of: child.value))
            parts.append("\(name):\(wrapper)")
        }

        return parts.sorted().joined(separator: "|")
    }
}

public actor HybridIndex<Doc: TantivyDocument> {
    private let baseURL: URL
    private let metadataURL: URL
    private let tantivyIndex: TantivyIndex
    private var hnswIndex: HnswIndex
    private let embeddingDimension: UInt32
    private let distanceType: HnswDistanceType
    private let hnswConfig: HnswIndexConfig
    private var nextDocId: UInt64
    private let primaryIdFieldName: String
    private let idFieldNames: [String]
    private let defaultTextFields: [String]
    private let schemaFingerprint: String

    public init(
        path: String,
        config: HybridIndexConfig = HybridIndexConfig(),
        primaryIdField: Doc.CodingKeys? = nil
    ) throws {
        let baseURL = URL(fileURLWithPath: path)
        let metadataURL = baseURL.appendingPathComponent(HybridConstants.metadataFileName)
        if FileManager.default.fileExists(atPath: metadataURL.path) {
            throw HybridSearchError.indexAlreadyExists
        }

        let idFields = HybridSchemaInspector.idFieldNames(for: Doc.self)
        guard !idFields.isEmpty else {
            throw HybridSearchError.missingIdField
        }

        let primaryName: String
        if let primary = primaryIdField {
            let name = primary.stringValue
            guard idFields.contains(name) else {
                throw HybridSearchError.invalidPrimaryIdField(name)
            }
            primaryName = name
        } else if idFields.count == 1 {
            primaryName = idFields[0]
        } else {
            throw HybridSearchError.ambiguousIdField(idFields)
        }

        let textFields = HybridSchemaInspector.textFieldNames(for: Doc.self)
        let defaultTextFields = textFields.filter { !idFields.contains($0) }

        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)

        let schemaBuilder = TantivySchemaExtractor.buildSchema(for: Doc.self)
        let docIdOptions = NumericFieldOptions(indexed: true, stored: true, fast: true, fieldnorms: false)
        schemaBuilder.addU64Field(name: HybridConstants.docIdFieldName, options: docIdOptions)

        let tantivyPath = baseURL.appendingPathComponent("tantivy")
        try FileManager.default.createDirectory(at: tantivyPath, withIntermediateDirectories: true)

        self.tantivyIndex = try TantivyIndex.newWithSchema(path: tantivyPath.path, schemaBuilder: schemaBuilder)
        self.hnswIndex = HnswIndex(
            maxConnections: config.hnswMaxConnections,
            maxElements: config.hnswMaxElements,
            maxLayers: config.hnswMaxLayers,
            efConstruction: config.hnswEfConstruction,
            dimension: config.embeddingDimension,
            distanceType: config.distanceType
        )

        self.baseURL = baseURL
        self.metadataURL = metadataURL
        self.embeddingDimension = config.embeddingDimension
        self.distanceType = config.distanceType
        self.hnswConfig = config.hnswConfig
        self.nextDocId = 0
        self.primaryIdFieldName = primaryName
        self.idFieldNames = idFields
        self.defaultTextFields = defaultTextFields
        self.schemaFingerprint = HybridSchemaInspector.schemaFingerprint(for: Doc.self)

        let metadata = HybridIndexMetadata(
            version: HybridIndexMetadata.currentVersion,
            embeddingDimension: config.embeddingDimension,
            distanceType: config.distanceType,
            hnswConfig: HybridHnswConfig(from: config.hnswConfig),
            nextDocId: nextDocId,
            primaryIdField: primaryName,
            schemaFingerprint: schemaFingerprint
        )
        try HybridMetadataStore.save(metadata, to: metadataURL)
    }

    public init(
        loadFrom path: String,
        primaryIdField: Doc.CodingKeys? = nil
    ) throws {
        let baseURL = URL(fileURLWithPath: path)
        let metadataURL = baseURL.appendingPathComponent(HybridConstants.metadataFileName)
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            throw HybridSearchError.metadataMissing
        }

        let metadata = try HybridMetadataStore.load(from: metadataURL)
        guard metadata.version == HybridIndexMetadata.currentVersion else {
            throw HybridSearchError.metadataCorrupt
        }

        let idFields = HybridSchemaInspector.idFieldNames(for: Doc.self)
        guard !idFields.isEmpty else {
            throw HybridSearchError.missingIdField
        }

        let primaryName: String
        if let primary = primaryIdField {
            let name = primary.stringValue
            guard idFields.contains(name) else {
                throw HybridSearchError.invalidPrimaryIdField(name)
            }
            primaryName = name
        } else {
            primaryName = metadata.primaryIdField
            guard idFields.contains(primaryName) else {
                throw HybridSearchError.invalidPrimaryIdField(primaryName)
            }
        }

        let currentFingerprint = HybridSchemaInspector.schemaFingerprint(for: Doc.self)
        guard currentFingerprint == metadata.schemaFingerprint else {
            throw HybridSearchError.metadataCorrupt
        }

        let textFields = HybridSchemaInspector.textFieldNames(for: Doc.self)
        let defaultTextFields = textFields.filter { !idFields.contains($0) }

        let schemaBuilder = TantivySchemaExtractor.buildSchema(for: Doc.self)
        let docIdOptions = NumericFieldOptions(indexed: true, stored: true, fast: true, fieldnorms: false)
        schemaBuilder.addU64Field(name: HybridConstants.docIdFieldName, options: docIdOptions)

        let tantivyPath = baseURL.appendingPathComponent("tantivy")
        self.tantivyIndex = try TantivyIndex.newWithSchema(path: tantivyPath.path, schemaBuilder: schemaBuilder)

        let hnswConfig = metadata.hnswConfig.toHnswIndexConfig(
            dimension: metadata.embeddingDimension,
            distanceType: metadata.distanceType
        )
        self.hnswIndex = try HnswIndex.load(
            directory: baseURL.path,
            basename: HybridConstants.hnswBasename,
            dimension: metadata.embeddingDimension,
            distanceType: metadata.distanceType,
            config: hnswConfig
        )

        self.baseURL = baseURL
        self.metadataURL = metadataURL
        self.embeddingDimension = metadata.embeddingDimension
        self.distanceType = metadata.distanceType
        self.hnswConfig = hnswConfig
        self.nextDocId = metadata.nextDocId
        self.primaryIdFieldName = primaryName
        self.idFieldNames = idFields
        self.defaultTextFields = defaultTextFields
        self.schemaFingerprint = currentFingerprint
    }

    public func getDimension() -> UInt32 {
        embeddingDimension
    }

    public func count() -> UInt64 {
        tantivyIndex.docsCount()
    }

    public func primaryIdField() -> String {
        primaryIdFieldName
    }

    @discardableResult
    public func add(doc: Doc, embedding: [Float]) async throws -> UInt64 {
        try validateEmbedding(embedding)

        let docId = nextDocId
        nextDocId += 1

        var fields = try doc.toTantivyDocument().fields
        fields.append(DocumentField(name: HybridConstants.docIdFieldName, value: .u64(docId)))

        try await hnswIndex.insert(vector: embedding, id: docId)

        do {
            try tantivyIndex.indexDoc(doc: TantivyDocumentFields(fields: fields))
        } catch {
            await hnswIndex.delete(id: docId)
            throw error
        }

        return docId
    }

    @discardableResult
    public func add(docs: [(Doc, [Float])]) async throws -> [UInt64] {
        guard !docs.isEmpty else {
            return []
        }

        for (_, embedding) in docs {
            try validateEmbedding(embedding)
        }

        let docIds: [UInt64] = (0..<docs.count).map { offset in
            nextDocId + UInt64(offset)
        }
        nextDocId += UInt64(docs.count)

        let nativeDocs: [TantivyDocumentFields] = try zip(docs, docIds).map { (pair, docId) in
            var fields = try pair.0.toTantivyDocument().fields
            fields.append(DocumentField(name: HybridConstants.docIdFieldName, value: .u64(docId)))
            return TantivyDocumentFields(fields: fields)
        }

        let embeddings = docs.map { $0.1 }
        try await hnswIndex.insertBatch(vectors: embeddings, ids: docIds)

        do {
            try tantivyIndex.indexDocs(docs: nativeDocs)
        } catch {
            await hnswIndex.delete(ids: docIds)
            throw error
        }

        return docIds
    }

    @discardableResult
    public func index(doc: Doc, embedding: [Float]) async throws -> UInt64 {
        let docId = try await add(doc: doc, embedding: embedding)
        try await commit()
        return docId
    }

    @discardableResult
    public func index(docs: [(Doc, [Float])]) async throws -> [UInt64] {
        let docIds = try await add(docs: docs)
        try await commit()
        return docIds
    }

    public func commit() async throws {
        try tantivyIndex.commit()
        try await hnswIndex.save(directory: baseURL.path, basename: HybridConstants.hnswBasename)
        try await hnswIndex.setSearchingMode(enabled: true)
        try persistMetadata()
    }

    public func delete(docId: UInt64, persist: Bool = true) async throws {
        let field = DocumentField(name: HybridConstants.docIdFieldName, value: .u64(docId))
        try tantivyIndex.deleteDoc(id: field)
        await hnswIndex.delete(id: docId)
        if persist {
            try await hnswIndex.save(directory: baseURL.path, basename: HybridConstants.hnswBasename)
            try persistMetadata()
        }
    }

    public func delete(idField: Doc.CodingKeys, idValue: FieldValue, persist: Bool = true) async throws {
        let field = DocumentField(name: idField.stringValue, value: idValue)
        guard let docFields = try? tantivyIndex.getDoc(id: field) else {
            return
        }
        guard let docId = docId(from: docFields) else {
            throw HybridSearchError.missingDocId
        }
        try await delete(docId: docId, persist: persist)
    }

    public func get(docId: UInt64) throws -> Doc? {
        let field = DocumentField(name: HybridConstants.docIdFieldName, value: .u64(docId))
        guard let docFields = try? tantivyIndex.getDoc(id: field) else {
            return nil
        }
        return try? Doc(fromFields: docFields)
    }

    public func get(idField: Doc.CodingKeys, idValue: FieldValue) throws -> Doc? {
        let field = DocumentField(name: idField.stringValue, value: idValue)
        guard let docFields = try? tantivyIndex.getDoc(id: field) else {
            return nil
        }
        return try? Doc(fromFields: docFields)
    }

    public func searchText(
        query: HybridTextQuery<Doc>,
        filter: TantivyQuery? = nil,
        limit: UInt32 = 10,
        offset: UInt32 = 0
    ) throws -> [HybridSearchResult<Doc>] {
        let hits = try bm25Search(query: query, filter: filter, limit: limit, offset: offset)
        return hits.compactMap { hit in
            guard let doc = try? Doc(fromFields: hit.fields) else { return nil }
            return HybridSearchResult(docId: hit.docId, score: hit.score, document: doc)
        }
    }

    public func searchVector(
        embedding: [Float],
        filter: TantivyQuery? = nil,
        limit: UInt32 = 10,
        offset: UInt32 = 0,
        efSearch: UInt32 = 100,
        overfetchMultiplier: UInt32 = 3
    ) async throws -> [HybridSearchResult<Doc>] {
        let hits = try await vectorSearchHits(
            embedding: embedding,
            filter: filter,
            limit: limit,
            offset: offset,
            efSearch: efSearch,
            overfetchMultiplier: overfetchMultiplier
        )

        let page = hits.dropFirst(Int(offset)).prefix(Int(limit))
        let docIds = page.map { $0.id }
        let docsById = try fetchDocs(for: docIds)

        return page.compactMap { hit in
            guard let doc = docsById[hit.id] else { return nil }
            return HybridSearchResult(
                docId: hit.id,
                score: similarity(from: hit.distance),
                document: doc
            )
        }
    }

    public func searchHybrid(
        query: HybridTextQuery<Doc>,
        embedding: [Float],
        filter: TantivyQuery? = nil,
        limit: UInt32 = 10,
        offset: UInt32 = 0,
        efSearch: UInt32 = 100,
        rrfK: Float = 60,
        textWeight: Float = 1,
        vectorWeight: Float = 1,
        overfetchMultiplier: UInt32 = 3
    ) async throws -> [HybridSearchResult<Doc>] {
        try validateEmbedding(embedding)

        let desired = max(1, limit + offset)
        let fetchLimit = max(1, desired * overfetchMultiplier)

        let bm25Hits = try bm25Search(query: query, filter: filter, limit: fetchLimit, offset: 0)
        let vectorHits = try await vectorSearchHits(
            embedding: embedding,
            filter: filter,
            limit: fetchLimit,
            offset: 0,
            efSearch: efSearch,
            overfetchMultiplier: 1
        )

        let ranked = rrfMerge(
            bm25: bm25Hits.map { $0.docId },
            vector: vectorHits.map { $0.id },
            rrfK: rrfK,
            textWeight: textWeight,
            vectorWeight: vectorWeight
        )

        let page = ranked.dropFirst(Int(offset)).prefix(Int(limit))
        let docIds = page.map { $0.docId }
        let docsById = try fetchDocs(for: docIds)

        return page.compactMap { item in
            guard let doc = docsById[item.docId] else { return nil }
            return HybridSearchResult(docId: item.docId, score: item.score, document: doc)
        }
    }

    public func compact() async throws {
        try await hnswIndex.compact(config: hnswConfig)
        try await hnswIndex.save(directory: baseURL.path, basename: HybridConstants.hnswBasename)
    }

    public func clear() async throws {
        try tantivyIndex.clearIndex()
        hnswIndex = HnswIndex(
            maxConnections: hnswConfig.maxNbConnection,
            maxElements: hnswConfig.maxElements,
            maxLayers: hnswConfig.maxLayer,
            efConstruction: hnswConfig.efConstruction,
            dimension: hnswConfig.dimension,
            distanceType: distanceType
        )
        nextDocId = 0
        try persistMetadata()
    }

    private func validateEmbedding(_ embedding: [Float]) throws {
        if embedding.count != Int(embeddingDimension) {
            throw HybridSearchError.dimensionMismatch(
                expected: embeddingDimension,
                got: UInt32(embedding.count)
            )
        }
    }

    private func persistMetadata() throws {
        let metadata = HybridIndexMetadata(
            version: HybridIndexMetadata.currentVersion,
            embeddingDimension: embeddingDimension,
            distanceType: distanceType,
            hnswConfig: HybridHnswConfig(from: hnswConfig),
            nextDocId: nextDocId,
            primaryIdField: primaryIdFieldName,
            schemaFingerprint: schemaFingerprint
        )
        try HybridMetadataStore.save(metadata, to: metadataURL)
    }

    private func bm25Search(
        query: HybridTextQuery<Doc>,
        filter: TantivyQuery?,
        limit: UInt32,
        offset: UInt32
    ) throws -> [(docId: UInt64, score: Float, fields: TantivyDocumentFields)] {
        let baseQuery = query.toTantivyQuery(defaultFieldsFallback: defaultTextFields)
        let combinedQuery = combine(query: baseQuery, filter: filter)
        let queryJson = try combinedQuery.toJson()
        let results = try tantivyIndex.searchDsl(
            queryJson: queryJson,
            topDocLimit: limit,
            topDocOffset: offset
        )

        return results.docs.compactMap { result in
            guard let docId = docId(from: result.doc) else { return nil }
            return (docId: docId, score: result.score, fields: result.doc)
        }
    }

    private func vectorSearchHits(
        embedding: [Float],
        filter: TantivyQuery?,
        limit: UInt32,
        offset: UInt32,
        efSearch: UInt32,
        overfetchMultiplier: UInt32
    ) async throws -> [HnswSearchResult] {
        try validateEmbedding(embedding)

        let desired = max(1, limit + offset)
        let fetchLimit = max(1, desired * overfetchMultiplier)
        let effectiveEf = max(efSearch, fetchLimit)

        try await hnswIndex.setSearchingMode(enabled: true)
        let results = try await hnswIndex.search(query: embedding, k: fetchLimit, efSearch: effectiveEf)

        guard let filter else {
            return results
        }

        let allowed = try filterDocIds(results.map { $0.id }, filter: filter)
        return results.filter { allowed.contains($0.id) }
    }

    private func filterDocIds(_ candidateIds: [UInt64], filter: TantivyQuery) throws -> Set<UInt64> {
        guard !candidateIds.isEmpty else {
            return []
        }

        let terms = candidateIds.map {
            TantivyQueryTerm(name: HybridConstants.docIdFieldName, value: .u64($0))
        }
        let idQuery = TantivyQuery.termSet(terms)
        let combined = TantivyQuery.boolean([
            TantivyBooleanClause(occur: .must, query: idQuery),
            TantivyBooleanClause(occur: .must, query: filter),
        ])
        let queryJson = try combined.toJson()
        let results = try tantivyIndex.searchDsl(
            queryJson: queryJson,
            topDocLimit: UInt32(candidateIds.count),
            topDocOffset: 0
        )

        var allowed: Set<UInt64> = []
        for result in results.docs {
            if let docId = docId(from: result.doc) {
                allowed.insert(docId)
            }
        }
        return allowed
    }

    private func fetchDocs(for docIds: [UInt64]) throws -> [UInt64: Doc] {
        guard !docIds.isEmpty else {
            return [:]
        }

        let fields = docIds.map { DocumentField(name: HybridConstants.docIdFieldName, value: .u64($0)) }
        let docs = try tantivyIndex.getDocsByIds(ids: fields)
        var mapped: [UInt64: Doc] = [:]

        for docFields in docs {
            guard let docId = docId(from: docFields) else { continue }
            if let doc = try? Doc(fromFields: docFields) {
                mapped[docId] = doc
            }
        }

        return mapped
    }

    private func docId(from fields: TantivyDocumentFields) -> UInt64? {
        TantivyDocumentFieldMap(fields).u64(HybridConstants.docIdFieldName)
    }

    private func combine(query: TantivyQuery, filter: TantivyQuery?) -> TantivyQuery {
        guard let filter else {
            return query
        }

        if case .all = query {
            return filter
        }

        return TantivyQuery.boolean([
            TantivyBooleanClause(occur: .must, query: query),
            TantivyBooleanClause(occur: .must, query: filter),
        ])
    }

    private func rrfMerge(
        bm25: [UInt64],
        vector: [UInt64],
        rrfK: Float,
        textWeight: Float,
        vectorWeight: Float
    ) -> [(docId: UInt64, score: Float)] {
        var scores: [UInt64: Float] = [:]

        for (rank, docId) in bm25.enumerated() {
            let score = textWeight / (rrfK + Float(rank + 1))
            scores[docId, default: 0] += score
        }

        for (rank, docId) in vector.enumerated() {
            let score = vectorWeight / (rrfK + Float(rank + 1))
            scores[docId, default: 0] += score
        }

        return scores
            .sorted { $0.value > $1.value }
            .map { (docId: $0.key, score: $0.value) }
    }

    private func similarity(from distance: Float) -> Float {
        1.0 / (1.0 + distance)
    }
}
