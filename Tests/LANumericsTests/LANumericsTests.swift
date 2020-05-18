import XCTest
import LANumerics
import simd

final class LANumericsTests: XCTestCase {

    public typealias LAFP = LANumeric & BinaryFloatingPoint
    
    func countingMatrix<F : BinaryFloatingPoint>(rows : Int, columns : Int) -> Matrix<F> {
        return Matrix<F>(rows: rows, columns: columns) { r, c in
            return F(c * rows + r + 1)
        }
    }
    
    func random<F : BinaryFloatingPoint>() -> F {
        F(Int.random(in: -100 ... 100))
    }
    
    func randomMatrix<F : BinaryFloatingPoint>(rows : Int = Int.random(in: 0 ... 10), columns : Int = Int.random(in: 0 ... 10)) -> Matrix<F> {
        return Matrix<F>(rows: rows, columns: columns) { _ , _ in
            random()
        }
    }
    
    func randomVector<F : BinaryFloatingPoint>(count : Int = Int.random(in: 0 ... 10)) -> Vector<F> {
        var X : Vector<F> = []
        for _ in 0 ..< count {
            X.append(random())
        }
        return X
    }
    
    func stress(iterations : Int = 1000, _ test : () -> Void) {
        for _ in 0 ..< iterations {
            test()
        }
    }

    func testTranspose() {
        let u : Matrix<Float> = countingMatrix(rows : 4, columns : 3)
        let v = u.transposed()
        XCTAssertEqual(u.rows, 4)
        XCTAssertEqual(u.columns, 3)
        XCTAssertEqual(v.rows, 3)
        XCTAssertEqual(v.columns, 4)
        for r in 0 ..< u.rows {
            for c in 0 ..< u.columns {
                XCTAssertEqual(u[r, c], v[c, r])
            }
        }
        XCTAssertEqual(v.transposed(), u)
        XCTAssertEqual(v, u′)
    }

    func testManhattanNorm() {
        func add(_ n : Int) -> Int { n * (n + 1) / 2 }
        func generic<E : LAFP>(_ type : E.Type) {
            let u : Matrix<E> = countingMatrix(rows : 4, columns : 3)
            XCTAssertEqual(u.manhattanNorm, E(add(u.rows * u.columns)))
            XCTAssertEqual(u.manhattanNorm, u.fold { x, y in x + abs(y) })
            let w : Matrix<E> = randomMatrix()
            XCTAssertEqual(w.manhattanNorm, w.fold { x, y in x + abs(y) })
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }

    func testEuclideanNorm() {
        func generic<E : LAFP>(_ type : E.Type) {
            let u : Matrix<E> = randomMatrix()
            let l2 = u.euclideanNorm
            let sum = u.fold { x, y in x + y*y }
            XCTAssertEqual((l2 * l2).rounded(), sum)
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }

    func testInfinityNorm() {
        func generic<E : LAFP>(_ type : E.Type) {
            let u : Matrix<E> = randomMatrix()
            let norm = u.infinityNorm
            let result = u.fold { x, y in max(x, abs(y)) }
            XCTAssertEqual(norm, result)
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }
    
    func testScaleAndAddFloat() {
        func generic<E : LAFP>(_ type : E.Type) {
            let m = Int.random(in: 1 ... 10)
            let n = Int.random(in: 1 ... 10)
            let u : Matrix<E> = randomMatrix(rows: m, columns: n)
            let v : Matrix<E> = randomMatrix(rows: m, columns: n)
            let alpha : E = random()
            let beta : E = random()
            let result = u.scaleAndAdd(alpha, beta, v)
            let spec = u.combine(v) { x, y in alpha * x + beta * y}
            XCTAssertEqual(result, spec)
            XCTAssertEqual(u + v, u.combine(v, { x, y in x + y }))
            XCTAssertEqual(u - v, u.combine(v, { x, y in x - y }))
            XCTAssertEqual(u + .zeros(m, n), u)
            XCTAssertEqual(.zeros(m, n) + u, u)
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }
    
    func testIndexOfLargestElement() {
        func generic<E : LAFP>(_ type : E.Type) {
            let u : Matrix<E> = randomMatrix()
            let largest = abs(u.largest)
            XCTAssert(u.forall { x in largest >= abs(x) }, "largest = \(largest) in \(u)")
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }
    
    func scale<E:LAFP>(_ alpha : E, _ vec : Vector<E>) -> Vector<E> {
        return vec.map { x in alpha * x }
    }

    func scale<E:LAFP>(_ alpha : E, _ matrix : Matrix<E>) -> Matrix<E> {
        return matrix.map { x in alpha * x }
    }
    
    func mul<E:LAFP>(_ A : Matrix<E>, _ B : Matrix<E>) -> Matrix<E> {
        let M = A.rows
        let N = B.columns
        let K = A.columns
        precondition(K == B.rows)
        var C = Matrix<E>(rows: M, columns: N)
        for m in 0 ..< M {
            for n in 0 ..< N {
                var sum : E = 0
                for k in 0 ..< K {
                    sum += A[m, k] * B[k, n]
                }
                C[m, n] = sum
            }
        }
        return C
    }

    func dot<E:LAFP>(_ X : Vector<E>, _ Y : Vector<E>) -> E {
        precondition(X.count == Y.count)
        var sum : E = 0
        for i in 0 ..< X.count {
            sum += X[i] * Y[i]
        }
        return sum
    }

    func testScale() {
        func generic<E : LAFP>(_ type : E.Type) {
            var X : Vector<E> = randomVector()
            let alpha : E = random()
            let Y = scale(alpha, X)
            E.scaleVector(alpha, &X)
            XCTAssertEqual(X, Y)
            X = []
            E.scaleVector(alpha, &X)
            XCTAssertEqual(X, [])
            let A : Matrix<E> = randomMatrix()
            XCTAssertEqual(alpha * A, scale(alpha, A))
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }
    
    func testDotProduct() {
        func generic<E : LAFP>(_ type : E.Type) {
            let X : Vector<E> = randomVector()
            let Y : Vector<E> = randomVector(count: X.count)
            XCTAssertEqual(E.dotProduct(X, Y), dot(X, Y))
            XCTAssertEqual(X ′* Y, dot(X, Y))
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }
    
    func testMatrixProduct() {
        func generic<E : LAFP>(_ type : E.Type) {
            let M = Int.random(in: 1 ... 10)
            let N = Int.random(in: 1 ... 10)
            let K = Int.random(in: 1 ... 10)
            let A : Matrix<E> = randomMatrix(rows: M, columns: K)
            let B : Matrix<E> = randomMatrix(rows: K, columns: N)
            XCTAssertEqual(A * B, mul(A, B))
            XCTAssertEqual(A′ ′* B, mul(A, B))
            XCTAssertEqual(A *′ B′, mul(A, B))
            XCTAssertEqual(A′ ′*′ B′, mul(A, B))
            let C : Matrix<E> = randomMatrix(rows: M, columns: N)
            let alpha : E = random()
            let beta : E = random()
            let R = scale(alpha, mul(A, B)) + scale(beta, C)
            func test(transposeA : Bool, transposeB : Bool) {
                let opA = transposeA ? A.transposed() : A
                let opB = transposeB ? B.transposed() : B
                var D = C
                E.matrixProduct(alpha, transposeA, opA, transposeB, opB, beta, &D)
                XCTAssertEqual(D, R)
            }
            test(transposeA: false, transposeB: false)
            test(transposeA: false, transposeB: true)
            test(transposeA: true, transposeB: false)
            test(transposeA: true, transposeB: true)
        }
        stress { generic(Float.self) }
        stress { generic(Double.self) }
    }

    func testMatrixVectorProduct() {
        func generic<E : LAFP>(_ type : E.Type) {
            let M = Int.random(in: 1 ... 10)
            let N = 1
            let K = Int.random(in: 1 ... 10)
            let A : Matrix<E> = randomMatrix(rows: M, columns: K)
            let B : Matrix<E> = randomMatrix(rows: K, columns: N)
            XCTAssertEqual(A * B.vector, mul(A, B).vector)
            XCTAssertEqual(A′ ′* B.vector, mul(A, B).vector)
            let C : Matrix<E> = randomMatrix(rows: M, columns: N)
            let alpha : E = random()
            let beta : E = random()
            let R = scale(alpha, mul(A, B)) + scale(beta, C)
            func test(transpose : Bool) {
                let opA = transpose ? A.transposed() : A
                let X = B.vector
                var Y = C.vector
                E.matrixVectorProduct(alpha, transpose, opA, X, beta, &Y)
                XCTAssertEqual(Y, R.vector)
            }
            test(transpose: false)
            test(transpose: true)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
        }
    }

    func testVectorVectorProduct() {
        func generic<E : LAFP>(_ type : E.Type) {
            let X : Matrix<E> = randomMatrix(rows: Int.random(in: 1 ... 10), columns: 1)
            let Y : Matrix<E> = randomMatrix(rows: 1, columns: Int.random(in: 1 ... 10))
            XCTAssertEqual(X.vector *′ Y.vector, mul(X, Y))
            var A : Matrix<E> = randomMatrix(rows: X.rows, columns: Y.columns)
            let alpha : E = random()
            let R = scale(alpha, mul(X, Y)) + A
            E.vectorVectorProduct(alpha, X.vector, Y.vector, &A)
            XCTAssertEqual(A, R)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
        }
    }
    
    func testSIMDVectors() {
        func generic<E : LAFP & SIMDScalar>(_ type : E.Type) {
            func test(_ count : Int, transform : (Matrix<E>) -> Matrix<E>) {
                let m : Matrix<E> = randomMatrix(rows: count, columns: 1)
                XCTAssertEqual(m, transform(m))
            }
            test(2) { m in Matrix(m.simd2) }
            test(2) { m in Matrix(row: m.simd2)′ }
            test(3) { m in Matrix(m.simd3) }
            test(3) { m in Matrix(row: m.simd3)′ }
            test(4) { m in Matrix(m.simd4) }
            test(4) { m in Matrix(row: m.simd4)′ }
            test(8) { m in Matrix(m.simd8) }
            test(8) { m in Matrix(row: m.simd8)′ }
            test(16) { m in Matrix(m.simd16) }
            test(16) { m in Matrix(row: m.simd16)′ }
            test(32) { m in Matrix(m.simd32) }
            test(32) { m in Matrix(row: m.simd32)′ }
            test(64) { m in Matrix(m.simd64) }
            test(64) { m in Matrix(row: m.simd64)′ }
        }
        stress {
            generic(Float.self)
            generic(Double.self)
        }
    }
    
    func testSIMDMatricesFloat() {
        typealias E = Float
        
        stress {
            let v2 : Matrix<E> = randomMatrix(rows: 2, columns: 1)
            let m2x2 : Matrix<E> = randomMatrix(rows: 2, columns: 2)
            XCTAssertEqual(m2x2, Matrix(m2x2.simd2x2))
            XCTAssertEqual((m2x2 * v2).simd2, m2x2.simd2x2 * v2.simd2)
            let m3x2 : Matrix<E> = randomMatrix(rows: 3, columns: 2)
            XCTAssertEqual(m3x2, Matrix(m3x2.simd2x3))
            XCTAssertEqual((m3x2 * v2).simd3, m3x2.simd2x3 * v2.simd2)
            let m4x2 : Matrix<E> = randomMatrix(rows: 4, columns: 2)
            XCTAssertEqual(m4x2, Matrix(m4x2.simd2x4))
            XCTAssertEqual((m4x2 * v2).simd4, m4x2.simd2x4 * v2.simd2)

            let v3 : Matrix<E> = randomMatrix(rows: 3, columns: 1)
            let m2x3 : Matrix<E> = randomMatrix(rows: 2, columns: 3)
            XCTAssertEqual(m2x3, Matrix(m2x3.simd3x2))
            XCTAssertEqual((m2x3 * v3).simd2, m2x3.simd3x2 * v3.simd3)
            let m3x3 : Matrix<E> = randomMatrix(rows: 3, columns: 3)
            XCTAssertEqual(m3x3, Matrix(m3x3.simd3x3))
            XCTAssertEqual((m3x3 * v3).simd3, m3x3.simd3x3 * v3.simd3)
            let m4x3 : Matrix<E> = randomMatrix(rows: 4, columns: 3)
            XCTAssertEqual(m4x3, Matrix(m4x3.simd3x4))
            XCTAssertEqual((m4x3 * v3).simd4, m4x3.simd3x4 * v3.simd3)

            let v4 : Matrix<E> = randomMatrix(rows: 4, columns: 1)
            let m2x4 : Matrix<E> = randomMatrix(rows: 2, columns: 4)
            XCTAssertEqual(m2x4, Matrix(m2x4.simd4x2))
            XCTAssertEqual((m2x4 * v4).simd2, m2x4.simd4x2 * v4.simd4)
            let m3x4 : Matrix<E> = randomMatrix(rows: 3, columns: 4)
            XCTAssertEqual(m3x4, Matrix(m3x4.simd4x3))
            XCTAssertEqual((m3x4 * v4).simd3, m3x4.simd4x3 * v4.simd4)
            let m4x4 : Matrix<E> = randomMatrix(rows: 4, columns: 4)
            XCTAssertEqual(m4x4, Matrix(m4x4.simd4x4))
            XCTAssertEqual((m4x4 * v4).simd4, m4x4.simd4x4 * v4.simd4)
        }
    }

    func testSIMDMatricesDouble() {
        typealias E = Double
        
        stress {
            let v2 : Matrix<E> = randomMatrix(rows: 2, columns: 1)
            let m2x2 : Matrix<E> = randomMatrix(rows: 2, columns: 2)
            XCTAssertEqual(m2x2, Matrix(m2x2.simd2x2))
            XCTAssertEqual((m2x2 * v2).simd2, m2x2.simd2x2 * v2.simd2)
            let m3x2 : Matrix<E> = randomMatrix(rows: 3, columns: 2)
            XCTAssertEqual(m3x2, Matrix(m3x2.simd2x3))
            XCTAssertEqual((m3x2 * v2).simd3, m3x2.simd2x3 * v2.simd2)
            let m4x2 : Matrix<E> = randomMatrix(rows: 4, columns: 2)
            XCTAssertEqual(m4x2, Matrix(m4x2.simd2x4))
            XCTAssertEqual((m4x2 * v2).simd4, m4x2.simd2x4 * v2.simd2)

            let v3 : Matrix<E> = randomMatrix(rows: 3, columns: 1)
            let m2x3 : Matrix<E> = randomMatrix(rows: 2, columns: 3)
            XCTAssertEqual(m2x3, Matrix(m2x3.simd3x2))
            XCTAssertEqual((m2x3 * v3).simd2, m2x3.simd3x2 * v3.simd3)
            let m3x3 : Matrix<E> = randomMatrix(rows: 3, columns: 3)
            XCTAssertEqual(m3x3, Matrix(m3x3.simd3x3))
            XCTAssertEqual((m3x3 * v3).simd3, m3x3.simd3x3 * v3.simd3)
            let m4x3 : Matrix<E> = randomMatrix(rows: 4, columns: 3)
            XCTAssertEqual(m4x3, Matrix(m4x3.simd3x4))
            XCTAssertEqual((m4x3 * v3).simd4, m4x3.simd3x4 * v3.simd3)

            let v4 : Matrix<E> = randomMatrix(rows: 4, columns: 1)
            let m2x4 : Matrix<E> = randomMatrix(rows: 2, columns: 4)
            XCTAssertEqual(m2x4, Matrix(m2x4.simd4x2))
            XCTAssertEqual((m2x4 * v4).simd2, m2x4.simd4x2 * v4.simd4)
            let m3x4 : Matrix<E> = randomMatrix(rows: 3, columns: 4)
            XCTAssertEqual(m3x4, Matrix(m3x4.simd4x3))
            XCTAssertEqual((m3x4 * v4).simd3, m3x4.simd4x3 * v4.simd4)
            let m4x4 : Matrix<E> = randomMatrix(rows: 4, columns: 4)
            XCTAssertEqual(m4x4, Matrix(m4x4.simd4x4))
            XCTAssertEqual((m4x4 * v4).simd4, m4x4.simd4x4 * v4.simd4)
        }
    }
    
    func testShape() {
        let a = Matrix<Float>.init(repeating: 1, rows: 1, columns: 2)
        let b = Matrix<Float>.init(repeating: 2, rows: 3, columns: 1)
        let c : Matrix<Float> = (1 / 100) * countingMatrix(rows: 4, columns: 3) + Matrix<Float>(repeating: 3, rows: 4, columns: 3)
        let d = Matrix<Float>.init(repeating: 4, rows: 2, columns: 2)
        let m = flatten(Matrix(rows: 2, columns: 2, elements: [a, c, b, d]))
        let col1 : Vector<Float> = [1.0, 0.0, 0.0, 3.01, 3.02, 3.03, 3.04]
        let col2 : Vector<Float> = [1.0, 0.0, 0.0, 3.05, 3.06, 3.07, 3.08]
        let col3 : Vector<Float> = [0.0, 0.0, 0.0, 3.09, 3.1, 3.11, 3.12]
        let col4 : Vector<Float> = [2.0, 2.0, 2.0, 4.0, 4.0]
        let col5 : Vector<Float> = [0.0, 0.0, 0.0, 4.0, 4.0]
        XCTAssertEqual(m, Matrix(columns: [col1, col2, col3, col4, col5]))
        XCTAssertEqual(m, Matrix(stackHorizontally: [Matrix(col1), Matrix(col2), Matrix(col3), Matrix(col4), Matrix(col5)]))
        let rows = (0 ..< m.rows).map { r in m.row(r).vector }
        XCTAssertEqual(m, Matrix(rows: rows))
        XCTAssertEqual(m, Matrix(stackVertically: (0 ..< m.rows).map { r in m.row(r) }))
    }
    
    func testSolveLinearEquations() {
        func generic<E : LAFP>(_ type : E.Type) {
            let A = Matrix<E>(rows: [[7, 5, -3], [3, -5, 2], [5, 3, -7]])
            let B = Matrix<E>([16, -8, 0])
            let X = Matrix<E>([1, 3, 2])
            XCTAssert((A.solve(B)! - X).infinityNorm < 0.0001)
            let Z = Matrix<E>(rows: 3, columns: 3)
            XCTAssertEqual(Z.solve(B), nil)
            XCTAssertEqual(Z.solve([0, 0, 0]), nil)
            XCTAssertEqual(A.solve([0, 0, 0])!, [0, 0, 0])
        }
        generic(Float.self)
        generic(Double.self)
    }

    func testSolveLinearLeastSquares() {
        func generic<E : LAFP>(_ type : E.Type) {
            let A = Matrix<E>(rows: [[7, 5, -3], [3, -5, 2], [5, 3, -7]])
            let B = Matrix<E>([16, -8, 0])
            let X = Matrix<E>([1, 3, 2])
            XCTAssert((A ∖ B - X).infinityNorm < 0.0001)
            XCTAssert((A′ ′∖ B - X).infinityNorm < 0.0001)
            let Z = Matrix<E>(rows: 3, columns: 3)
            XCTAssertEqual(Z ∖ B, Matrix<E>.zeros(3, 1))
            XCTAssertEqual(Z ∖ [0, 0, 0], [0, 0, 0])
            XCTAssertEqual(A ∖ [0, 0, 0], [0, 0, 0])
            XCTAssertEqual(Matrix<E>() ∖ [], [])
        }
        generic(Float.self)
        generic(Double.self)
    }
    
    func testLaeuchliExample() {
        func generic<E : LAFP>(eps : E) {
            let A = Matrix(rows: [[1, 1], [eps, 0], [0, eps]])
            let b = [2, eps, eps]
            XCTAssertNil((A′*A).solve(A′*b))
            XCTAssert((A ∖ Matrix(b) - Matrix([1, 1])).infinityNorm < 10 * eps)
            XCTAssert((A′ ′∖ Matrix(b) - Matrix([1, 1])).infinityNorm < 10 * eps)
        }
        generic(eps: Double(1e-16))
        generic(eps: Float(1e-7))
    }

    
}
