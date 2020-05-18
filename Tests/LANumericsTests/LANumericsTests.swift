import XCTest
@testable import LANumerics

final class LANumericsTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(LANumerics().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
