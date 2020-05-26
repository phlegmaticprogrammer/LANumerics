import XCTest
import LANumerics
import Numerics
import simd

final class LANumericsTests: XCTestCase {

    public typealias Num = LANumeric

    func countingMatrix<F : Num>(rows : Int, columns : Int) -> Matrix<F> {
        return Matrix<F>(rows: rows, columns: columns) { r, c in
            return F(exactly: c * rows + r + 1)!
        }
    }
    
    func randomWhole<F : Num>() -> F {
        F.randomWhole(in: -100 ... 100)
    }
    
    func randomWholeMatrix<F : Num>(rows : Int = Int.random(in: 0 ... 10), columns : Int = Int.random(in: 0 ... 10)) -> Matrix<F> {
        return Matrix<F>(rows: rows, columns: columns) { _ , _ in
            randomWhole()
        }
    }
    
    func randomWholeVector<F : Num>(count : Int = Int.random(in: 0 ... 10)) -> Vector<F> {
        var X : Vector<F> = []
        for _ in 0 ..< count {
            X.append(randomWhole())
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
        let v = u.transpose
        XCTAssertEqual(u.rows, 4)
        XCTAssertEqual(u.columns, 3)
        XCTAssertEqual(v.rows, 3)
        XCTAssertEqual(v.columns, 4)
        for r in 0 ..< u.rows {
            for c in 0 ..< u.columns {
                XCTAssertEqual(u[r, c], v[c, r])
            }
        }
        XCTAssertEqual(v.transpose, u)
        XCTAssertEqual(v, u′)
    }

    func testManhattanNorm() {
        func add(_ n : Int) -> Int { n * (n + 1) / 2 }
        func generic<E : Num>(_ type : E.Type) {
            let u : Matrix<E> = countingMatrix(rows : 4, columns : 3)
            XCTAssertEqual(u.manhattanNorm, E.Magnitude(exactly: add(u.rows * u.columns))!)
            XCTAssertEqual(u.manhattanNorm, u.reduce(0) { x, y in x + y.manhattanLength })
            let w : Matrix<E> = randomWholeMatrix()
            XCTAssertEqual(w.manhattanNorm, w.reduce(0) { x, y in x + y.manhattanLength })
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testNorm() {
        func generic<E : Num>(_ type : E.Type) {
            let u : Matrix<E> = randomWholeMatrix()
            let l2 = u.norm
            let sum = u.reduce(0) { x, y in x + y.lengthSquared }
            XCTSame(l2 * l2, sum)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testScaleAndAddFloat() {
        func generic<E : Num>(_ type : E.Type) {
            let m = Int.random(in: 1 ... 10)
            let n = Int.random(in: 1 ... 10)
            let u : Matrix<E> = randomWholeMatrix(rows: m, columns: n)
            let v : Matrix<E> = randomWholeMatrix(rows: m, columns: n)
            let alpha : E = randomWhole()
            let beta : E = randomWhole()
            var result = u
            result.accumulate(alpha, beta, v)
            let spec = u.combine(v) { x, y in alpha * x + beta * y}
            XCTAssertEqual(result, spec)
            XCTAssertEqual(u + v, u.combine(v, { x, y in x + y }))
            XCTAssertEqual(u - v, u.combine(v, { x, y in x - y }))
            XCTAssertEqual(u + .zeros(m, n), u)
            XCTAssertEqual(.zeros(m, n) + u, u)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testMaxNorm() {
        func generic<E : Num>(_ type : E.Type) {
            let u : Matrix<E> = randomWholeMatrix()
            let norm = u.maxNorm
            XCTAssert(u.forall { x in norm >= x.manhattanLength }, "norm = \(norm) of \(u)")
            XCTAssert((u.vector.isEmpty && norm == 0) || u.exists { x in norm == x.manhattanLength }, "norm = \(norm) of \(u)")
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testInfNorm() {
        func generic<E : Num>(_ type : E.Type) {
            let u : Matrix<E> = randomWholeMatrix()
            let norm = u.infNorm
            XCTAssert(u.forall { x in norm >= x.magnitude }, "norm = \(norm) of \(u)")
            XCTAssert((u.vector.isEmpty && norm == 0) || u.exists { x in norm == x.magnitude }, "norm = \(norm) of \(u)")
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func scale<E:Num>(_ alpha : E, _ vec : Vector<E>) -> Vector<E> {
        return vec.map { x in alpha * x }
    }

    func scale<E:Num>(_ alpha : E, _ matrix : Matrix<E>) -> Matrix<E> {
        return matrix.map { x in alpha * x }
    }
    
    func mul<E:Num>(_ A : Matrix<E>, _ B : Matrix<E>) -> Matrix<E> {
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

    func dot<E:Num>(_ X : Vector<E>, _ Y : Vector<E>) -> E {
        precondition(X.count == Y.count)
        var sum : E = 0
        for i in 0 ..< X.count {
            sum += X[i] * Y[i]
        }
        return sum
    }
    
    func epsilon<E : Num>(_ type : E.Type) -> E.Magnitude {
        let e : E = 0.1
        return e.magnitude
    }
    
    func XCTSame<E : Num>(_ X : Matrix<E>, _ Y : Matrix<E>) {
        let norm = (X - Y).maxNorm
        XCTAssert(norm < epsilon(E.self), "norm is \(norm), X = \(X), Y = \(Y)")
    }

    func XCTSame<E : Num>(_ X : E, _ Y : E) {
        let norm = (X - Y).magnitude
        let e : E.Magnitude = epsilon(E.self)
        XCTAssert(norm < e, "norm is \(norm), X = \(X), Y = \(Y)")
    }

    func XCTSame<E : Num>(_ X : Vector<E>, _ Y : Vector<E>) {
        XCTSame(Matrix(X), Matrix(Y))
    }
    
    func testScale() {
        func generic<E : Num>(_ type : E.Type) {
            var X : Vector<E> = randomWholeVector()
            let alpha : E = randomWhole()
            let Y = scale(alpha, X)
            E.scaleVector(alpha, &X)
            XCTAssertEqual(X, Y)
            X = []
            E.scaleVector(alpha, &X)
            XCTAssertEqual(X, [])
            let A : Matrix<E> = randomWholeMatrix()
            XCTAssertEqual(alpha * A, scale(alpha, A))
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testDotProduct() {
        func generic<E : Num>(_ type : E.Type) {
            let X : Vector<E> = randomWholeVector()
            let Y : Vector<E> = randomWholeVector(count: X.count)
            XCTAssertEqual(E.dotProduct(X, Y), dot(X, Y))
            XCTAssertEqual(X * Y, dot(X, Y))
            XCTAssertEqual(E.adjointDotProduct(X, Y), dot(X′.vector, Y))
            XCTAssertEqual(X ′* Y, dot(X′.vector, Y))
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testMatrixProduct() {
        func generic<E : Num>(_ type : E.Type) {
            let M = Int.random(in: 0 ... 10)
            let N = Int.random(in: 0 ... 10)
            let K = Int.random(in: 0 ... 10)
            let A : Matrix<E> = randomWholeMatrix(rows: M, columns: K)
            let B : Matrix<E> = randomWholeMatrix(rows: K, columns: N)
            XCTAssertEqual(A * B, mul(A, B))
            XCTAssertEqual(A′ ′* B, mul(A, B))
            XCTAssertEqual(A *′ B′, mul(A, B))
            XCTAssertEqual(A′ ′*′ B′, mul(A, B))
            let C : Matrix<E> = randomWholeMatrix(rows: M, columns: N)
            let alpha : E = randomWhole()
            let beta : E = randomWhole()
            let R = scale(alpha, mul(A, B)) + scale(beta, C)
            func test(transposeA : Transpose, transposeB : Transpose) {
                let opA = transposeA.apply(A)
                let opB = transposeB.apply(B)
                var D = C
                E.matrixProduct(alpha, transposeA, opA, transposeB, opB, beta, &D)
                XCTAssertEqual(D, R)
            }
            test(transposeA: .none, transposeB: .none)
            test(transposeA: .transpose, transposeB: .none)
            test(transposeA: .adjoint, transposeB: .none)
            test(transposeA: .none, transposeB: .transpose)
            test(transposeA: .transpose, transposeB: .transpose)
            test(transposeA: .adjoint, transposeB: .transpose)
            test(transposeA: .none, transposeB: .adjoint)
            test(transposeA: .transpose, transposeB: .adjoint)
            test(transposeA: .adjoint, transposeB: .adjoint)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testMatrixVectorProduct() {
        func generic<E : Num>(_ type : E.Type) {
            let M = Int.random(in: 0 ... 10)
            let N = 1
            let K = Int.random(in: 0 ... 10)
            let A : Matrix<E> = randomWholeMatrix(rows: M, columns: K)
            let B : Matrix<E> = randomWholeMatrix(rows: K, columns: N)
            XCTAssertEqual(A * B.vector, mul(A, B).vector)
            XCTAssertEqual(A′ ′* B.vector, mul(A, B).vector)
            let C : Matrix<E> = randomWholeMatrix(rows: M, columns: N)
            let alpha : E = randomWhole()
            let beta : E = randomWhole()
            let R = scale(alpha, mul(A, B)) + scale(beta, C)
            func test(transpose : Transpose) {
                let opA = transpose.apply(A)
                let X = B.vector
                var Y = C.vector
                E.matrixVectorProduct(alpha, transpose, opA, X, beta, &Y)
                XCTAssertEqual(Y, R.vector)
            }
            test(transpose: .none)
            test(transpose: .transpose)
            test(transpose: .adjoint)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testVectorAdjointVectorProduct() {
        func generic<E : Num>(_ type : E.Type) {
            let X : Matrix<E> = randomWholeMatrix(rows: Int.random(in: 0 ... 10), columns: 1)
            let Y : Matrix<E> = randomWholeMatrix(rows: Int.random(in: 0 ... 10), columns: 1)
            XCTAssertEqual(X.vector *′ Y.vector, mul(X, Y′))
            var A : Matrix<E> = randomWholeMatrix(rows: X.rows, columns: Y.rows)
            let alpha : E = randomWhole()
            let R = scale(alpha, mul(X, Y′)) + A
            E.vectorAdjointVectorProduct(alpha, X.vector, Y.vector, &A)
            XCTAssertEqual(A, R)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testVectorVectorProduct() {
        func generic<E : Num>(_ type : E.Type) {
            let X : Matrix<E> = randomWholeMatrix(rows: Int.random(in: 0 ... 10), columns: 1)
            let Y : Matrix<E> = randomWholeMatrix(rows: Int.random(in: 0 ... 10), columns: 1)
            var A : Matrix<E> = randomWholeMatrix(rows: X.rows, columns: Y.rows)
            let alpha : E = randomWhole()
            let R = scale(alpha, mul(X, Y.transpose)) + A
            E.vectorVectorProduct(alpha, X.vector, Y.vector, &A)
            XCTAssertEqual(A, R)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testSIMDVectors() {
        func generic<E : Num & SIMDScalar>(_ type : E.Type) {
            func test(_ count : Int, transform : (Matrix<E>) -> Matrix<E>) {
                let m : Matrix<E> = randomWholeMatrix(rows: count, columns: 1)
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
            let v2 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 1)
            let m2x2 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 2)
            XCTAssertEqual(m2x2, Matrix(m2x2.simd2x2))
            XCTAssertEqual((m2x2 * v2).simd2, m2x2.simd2x2 * v2.simd2)
            let m3x2 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 2)
            XCTAssertEqual(m3x2, Matrix(m3x2.simd2x3))
            XCTAssertEqual((m3x2 * v2).simd3, m3x2.simd2x3 * v2.simd2)
            let m4x2 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 2)
            XCTAssertEqual(m4x2, Matrix(m4x2.simd2x4))
            XCTAssertEqual((m4x2 * v2).simd4, m4x2.simd2x4 * v2.simd2)

            let v3 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 1)
            let m2x3 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 3)
            XCTAssertEqual(m2x3, Matrix(m2x3.simd3x2))
            XCTAssertEqual((m2x3 * v3).simd2, m2x3.simd3x2 * v3.simd3)
            let m3x3 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 3)
            XCTAssertEqual(m3x3, Matrix(m3x3.simd3x3))
            XCTAssertEqual((m3x3 * v3).simd3, m3x3.simd3x3 * v3.simd3)
            let m4x3 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 3)
            XCTAssertEqual(m4x3, Matrix(m4x3.simd3x4))
            XCTAssertEqual((m4x3 * v3).simd4, m4x3.simd3x4 * v3.simd3)

            let v4 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 1)
            let m2x4 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 4)
            XCTAssertEqual(m2x4, Matrix(m2x4.simd4x2))
            XCTAssertEqual((m2x4 * v4).simd2, m2x4.simd4x2 * v4.simd4)
            let m3x4 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 4)
            XCTAssertEqual(m3x4, Matrix(m3x4.simd4x3))
            XCTAssertEqual((m3x4 * v4).simd3, m3x4.simd4x3 * v4.simd4)
            let m4x4 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 4)
            XCTAssertEqual(m4x4, Matrix(m4x4.simd4x4))
            XCTAssertEqual((m4x4 * v4).simd4, m4x4.simd4x4 * v4.simd4)
        }
    }

    func testSIMDMatricesDouble() {
        typealias E = Double
        
        stress {
            let v2 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 1)
            let m2x2 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 2)
            XCTAssertEqual(m2x2, Matrix(m2x2.simd2x2))
            XCTAssertEqual((m2x2 * v2).simd2, m2x2.simd2x2 * v2.simd2)
            let m3x2 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 2)
            XCTAssertEqual(m3x2, Matrix(m3x2.simd2x3))
            XCTAssertEqual((m3x2 * v2).simd3, m3x2.simd2x3 * v2.simd2)
            let m4x2 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 2)
            XCTAssertEqual(m4x2, Matrix(m4x2.simd2x4))
            XCTAssertEqual((m4x2 * v2).simd4, m4x2.simd2x4 * v2.simd2)

            let v3 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 1)
            let m2x3 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 3)
            XCTAssertEqual(m2x3, Matrix(m2x3.simd3x2))
            XCTAssertEqual((m2x3 * v3).simd2, m2x3.simd3x2 * v3.simd3)
            let m3x3 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 3)
            XCTAssertEqual(m3x3, Matrix(m3x3.simd3x3))
            XCTAssertEqual((m3x3 * v3).simd3, m3x3.simd3x3 * v3.simd3)
            let m4x3 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 3)
            XCTAssertEqual(m4x3, Matrix(m4x3.simd3x4))
            XCTAssertEqual((m4x3 * v3).simd4, m4x3.simd3x4 * v3.simd3)

            let v4 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 1)
            let m2x4 : Matrix<E> = randomWholeMatrix(rows: 2, columns: 4)
            XCTAssertEqual(m2x4, Matrix(m2x4.simd4x2))
            XCTAssertEqual((m2x4 * v4).simd2, m2x4.simd4x2 * v4.simd4)
            let m3x4 : Matrix<E> = randomWholeMatrix(rows: 3, columns: 4)
            XCTAssertEqual(m3x4, Matrix(m3x4.simd4x3))
            XCTAssertEqual((m3x4 * v4).simd3, m3x4.simd4x3 * v4.simd4)
            let m4x4 : Matrix<E> = randomWholeMatrix(rows: 4, columns: 4)
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
        func generic<E : Num>(_ type : E.Type) {
            let A = Matrix<E>(rows: [[7, 5, -3], [3, -5, 2], [5, 3, -7]])
            let B = Matrix<E>([16, -8, 0])
            let X = Matrix<E>([1, 3, 2])
            let eps = epsilon(E.self)
            XCTAssert((A.solve(B)! - X).maxNorm < eps)
            let Z = Matrix<E>(rows: 3, columns: 3)
            XCTAssertEqual(Z.solve(B), nil)
            XCTAssertEqual(Z.solve([0, 0, 0]), nil)
            XCTAssertEqual(A.solve([0, 0, 0])!, [0, 0, 0])
        }
        generic(Float.self)
        generic(Double.self)
        generic(Complex<Float>.self)
        generic(Complex<Double>.self)
    }

    func testSolveLinearLeastSquares() {
        func generic<E : Num>(_ type : E.Type) {
            let A = Matrix<E>(rows: [[7, 5, -3], [3, -5, 2], [5, 3, -7]])
            let B = Matrix<E>([16, -8, 0])
            let X = Matrix<E>([1, 3, 2])
            let eps = epsilon(E.self)
            XCTAssert((A ∖ B - X).maxNorm < eps)
            XCTAssert((A′ ′∖ B - X).maxNorm < eps)
            let Z = Matrix<E>(rows: 3, columns: 3)
            XCTAssertEqual(Z ∖ B, Matrix<E>.zeros(3, 1))
            XCTAssertEqual(Z ∖ [0, 0, 0], [0, 0, 0])
            XCTAssertEqual(A ∖ [0, 0, 0], [0, 0, 0])
            XCTAssertEqual(Matrix<E>() ∖ [], [])
        }
        generic(Float.self)
        generic(Double.self)
        generic(Complex<Float>.self)
        generic(Complex<Double>.self)
    }
    
    func testLäuchliExample() {
        func generic<E : Num & BinaryFloatingPoint>(eps : E) {
            let A = Matrix(rows: [[1, 1], [eps, 0], [0, eps]])
            let b = [2, eps, eps]
            XCTAssertNil((A′*A).solve(A′*b))
            XCTAssert((A ∖ Matrix(b) - Matrix([1, 1])).maxNorm < 10 * eps)
            XCTAssert((A′ ′∖ Matrix(b) - Matrix([1, 1])).maxNorm < 10 * eps)
        }
        generic(eps: Double(1e-16))
        generic(eps: Float(1e-7))
    }
    
    func testSingularValueDecomposition() {
        func generic<E : Num>(_ type : E.Type) {
            func same(_ X : Matrix<E>, _ Y : Matrix<E>) {
                let norm = (X - Y).maxNorm
                XCTAssert(norm < epsilon(E.self), "norm is \(norm), X = \(X), Y = \(Y)")
            }
            func isSame(_ X : Matrix<E>, _ Y : Matrix<E>) -> Bool {
                let norm = (X - Y).maxNorm
                return norm < epsilon(E.self)
            }
            let A : Matrix<E> = randomWholeMatrix()
            let m = A.rows
            let n = A.columns
            let k = min(m, n)
            let svd = A.svd()!
            XCTAssertEqual(svd.singularValues.count, k)
            XCTAssert(svd.left.hasDimensions(m, m))
            XCTAssert(svd.right.hasDimensions(n, n))
            same(svd.left ′* svd.left, .eye(m))
            same(svd.right ′* svd.right, .eye(n))
            func map(_ v : Vector<E.Magnitude>) -> Vector<E> {
                v.map { x in E(magnitude: x) }
            }
            let D = Matrix<E>(rows: A.rows, columns: A.columns, diagonal: map(svd.singularValues))
            same(svd.left * D * svd.right, A)
            let svdS = A.svd(left: .singular, right: .singular)!
            same(Matrix(map(svd.singularValues)), Matrix(map(svdS.singularValues)))
            same(svd.left[0 ..< m, 0 ..< k], svdS.left)
            same(svd.right[0 ..< k, 0 ..< n], svdS.right)
            let svdN = A.svd(left: .none, right: .none)!
            same(Matrix(map(svd.singularValues)), Matrix(map(svdN.singularValues)))
            same(svd.left[0 ..< m, 0 ..< 0], svdN.left)
            same(svd.right[0 ..< 0, 0 ..< n], svdN.right)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func test_blas_asum() {
        func generic<E : Num>(_ type : E.Type) {
            let v : Vector<E> = randomWholeVector()
            let asum = Matrix(v).reduce(0) { x, y in x + y.manhattanLength }
            let blas_asum = E.blas_asum(Int32(v.count), v, 1)
            XCTAssertEqual(asum, blas_asum)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func test_blas_nrm2() {
        func generic<E : Num>(_ type : E.Type) {
            let v : Vector<E> = randomWholeVector()
            let nrm2_squared = Matrix(v).reduce(0) { x, y in x + y.lengthSquared }
            let blas_nrm2 = E.blas_nrm2(Int32(v.count), v, 1)
            XCTSame(E(magnitude: nrm2_squared), E(magnitude: blas_nrm2 * blas_nrm2))
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
        
    func test_blas_scal() {
        func generic<E : Num>(_ type : E.Type) {
            var v : Vector<E> = randomWholeVector()
            let alpha : E = randomWhole()
            let result = scale(alpha, v)
            E.blas_scal(Int32(v.count), alpha, &v, 1)
            XCTAssertEqual(v, result)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func testEigen() {
        func generic<E : Num>(_ type : E.Type) {
            let n = Int.random(in: 0 ... 10)
            var A : Matrix<E> = randomWholeMatrix(rows: n, columns: n)
            A = 0.5 * (A + A′)
            let eigen = A.eigen()!
            let D = Matrix(diagonal: eigen.eigenValues.map{ x in E(magnitude: x) })
            let B = eigen.eigenVectors
            XCTSame(B ′* B, .eye(n))
            XCTSame(B * D *′ B, A)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func testSchur() {
        func generic<E : Num>(_ type : E.Type) where E.Magnitude : Real {
            let n = Int.random(in: 0 ... 10)
            let A : Matrix<E> = randomWholeMatrix(rows: n, columns: n)
            let schur = A.schur()!
            let D = schur.schurForm
            let B = schur.schurVectors
            XCTSame(B ′* B, .eye(n))
            XCTSame(B * D *′ B, A)
            if E.Magnitude.self == E.self {
                XCTAssert(D.isQuasiUpperTriangle)
            } else {
                XCTAssert(D.isUpperTriangle)
            }
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func test_vDSP_convert() {
        func generic<R : Real & LANumeric>(_ type : R.Type) {
            let v : Vector<Complex<R>> = randomWholeVector()
            let real = v.map { x in x.real }
            let imaginary = v.map { x in x.imaginary }
            let split = Complex<R>.vDSP_convert(interleavedComplex: v)
            XCTAssertEqual(real, split.real)
            XCTAssertEqual(imaginary, split.imaginary)
            let u = Complex<R>.vDSP_convert(real: real, imaginary: imaginary)
            XCTAssertEqual(u, v)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
        }
    }
    
    func test_vDSP_elementwise_absolute() {
        func generic<E : Num>(_ type : E.Type) {
            let v : [E] = randomWholeVector()
            let abs = E.vDSP_elementwise_absolute(v)
            let vabs = v.map { x in x.length }
            XCTSame(abs, vabs)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func test_vDSP_elementwise_adjoint() {
        func generic<E : Num>(_ type : E.Type) {
            let v : [E] = randomWholeVector()
            let abs = E.vDSP_elementwise_adjoint(v)
            let vabs = v.map { x in x.adjoint }
            XCTSame(abs, vabs)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func test_vDSP_elementwise_multiply() {
        func generic<E : Num>(_ type : E.Type) {
            let u : [E] = randomWholeVector()
            let v : [E] = randomWholeVector(count: u.count)
            let result = E.vDSP_elementwise_multiply(u, v)
            let predicted = zip(u, v).map { x, y in x * y }
            XCTSame(result, predicted)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

    func test_vDSP_elementwise_divide() {
        func generic<E : Num>(_ type : E.Type) {
            let u : [E] = randomWholeVector()
            let v : [E] = randomWholeVector(count: u.count).map { x in x == 0 ? 1 : x }
            let result = E.vDSP_elementwise_divide(u, v)
            let predicted = zip(u, v).map { x, y in x / y }
            XCTSame(result, predicted)
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }
    
    func test_vDSP_elementwise_operators() {
        func generic<E : Num>(_ type : E.Type) {
            let u : [E] = randomWholeVector()
            let v : [E] = randomWholeVector(count: u.count).map { x in x == 0 ? 1 : x }
            let u_plus_v = zip(u, v).map { x, y in x + y }
            XCTAssertEqual(u .+ v, u_plus_v)
            XCTAssertEqual(Matrix(u) .+ Matrix(v), Matrix(u_plus_v))
            let u_minus_v = zip(u, v).map { x, y in x - y }
            XCTAssertEqual(u .- v, u_minus_v)
            XCTAssertEqual(Matrix(u) .- Matrix(v), Matrix(u_minus_v))
            let u_times_v = zip(u, v).map { x, y in x * y }
            XCTAssertEqual(u .* v, u_times_v)
            XCTAssertEqual(Matrix(u) .* Matrix(v), Matrix(u_times_v))
            let u_div_v = zip(u, v).map { x, y in x / y }
            XCTSame(u ./ v, u_div_v)
            XCTSame(Matrix(u) ./ Matrix(v), Matrix(u_div_v))
        }
        stress {
            generic(Float.self)
            generic(Double.self)
            generic(Complex<Float>.self)
            generic(Complex<Double>.self)
        }
    }

}
