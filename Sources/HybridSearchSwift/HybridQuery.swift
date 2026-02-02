import Foundation
import TantivySwift

public struct HybridTextQuery<Doc: TantivySearchableDocument & Sendable>: Sendable {
    public var query: String
    public var defaultFields: [Doc.CodingKeys]
    public var fuzzyFields: [TantivySwiftFuzzyField<Doc>]

    public init(
        query: String,
        defaultFields: [Doc.CodingKeys] = [],
        fuzzyFields: [TantivySwiftFuzzyField<Doc>] = []
    ) {
        self.query = query
        self.defaultFields = defaultFields
        self.fuzzyFields = fuzzyFields
    }
}

private extension TantivySwiftFuzzyField {
    func toQueryFuzzyField() -> TantivyQueryFuzzyField {
        TantivyQueryFuzzyField(
            fieldName: field.stringValue,
            prefix: prefix,
            distance: distance,
            transposeCostOne: transposeCostOne
        )
    }
}

extension HybridTextQuery {
    func toTantivyQuery(defaultFieldsFallback: [String]) -> TantivyQuery {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return .all
        }

        let fields = defaultFields.isEmpty ? defaultFieldsFallback : defaultFields.map { $0.stringValue }
        let fuzzy = fuzzyFields.map { $0.toQueryFuzzyField() }
        let queryString = TantivyQueryString(query: query, defaultFields: fields, fuzzyFields: fuzzy)
        return .queryString(queryString)
    }
}
