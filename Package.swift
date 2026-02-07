// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HybridSearchSwift",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
    ],
    products: [
        .library(
            name: "HybridSearchSwift",
            targets: ["HybridSearchSwift"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/botisan-ai/tantivy.swift.git", from: "0.3.2"),
        .package(url: "https://github.com/botisan-ai/HNSW.swift.git", from: "0.2.2"),
    ],
    targets: [
        .target(
            name: "HybridSearchSwift",
            dependencies: [
                .product(name: "TantivySwift", package: "tantivy.swift"),
                .product(name: "HnswSwift", package: "HNSW.swift"),
            ]
        ),
        .testTarget(
            name: "HybridSearchSwiftTests",
            dependencies: ["HybridSearchSwift"]
        ),
    ]
)
