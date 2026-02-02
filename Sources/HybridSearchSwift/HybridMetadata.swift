import Foundation
import HnswSwift

struct HybridHnswConfig: Codable, Sendable {
    let maxConnections: UInt32
    let maxElements: UInt64
    let maxLayers: UInt32
    let efConstruction: UInt32

    init(maxConnections: UInt32, maxElements: UInt64, maxLayers: UInt32, efConstruction: UInt32) {
        self.maxConnections = maxConnections
        self.maxElements = maxElements
        self.maxLayers = maxLayers
        self.efConstruction = efConstruction
    }

    init(from config: HnswIndexConfig) {
        self.maxConnections = config.maxNbConnection
        self.maxElements = config.maxElements
        self.maxLayers = config.maxLayer
        self.efConstruction = config.efConstruction
    }

    func toHnswIndexConfig(dimension: UInt32, distanceType: HnswDistanceType) -> HnswIndexConfig {
        HnswIndexConfig(
            maxConnections: maxConnections,
            maxElements: maxElements,
            maxLayers: maxLayers,
            efConstruction: efConstruction,
            dimension: dimension,
            distanceType: distanceType
        )
    }
}

struct HybridIndexMetadata: Codable, Sendable {
    static let currentVersion = 1

    let version: Int
    let embeddingDimension: UInt32
    let distanceType: HnswDistanceType
    let hnswConfig: HybridHnswConfig
    let nextDocId: UInt64
    let primaryIdField: String
    let schemaFingerprint: String
}

enum HybridMetadataStore {
    static func save(_ metadata: HybridIndexMetadata, to url: URL) throws {
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: url, options: .atomic)
    }

    static func load(from url: URL) throws -> HybridIndexMetadata {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(HybridIndexMetadata.self, from: data)
    }
}

extension HnswDistanceType: @retroactive Codable {
    private enum CodingError: Error {
        case invalidValue(String)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(stringValue)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        switch value {
        case "l2":
            self = .l2
        case "cosine":
            self = .cosine
        case "dot":
            self = .dot
        case "l1":
            self = .l1
        default:
            throw CodingError.invalidValue("Unknown distance type: \(value)")
        }
    }

    var stringValue: String {
        switch self {
        case .l2: return "l2"
        case .cosine: return "cosine"
        case .dot: return "dot"
        case .l1: return "l1"
        }
    }
}
