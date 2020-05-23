import Accelerate
import Numerics

public enum SVDJob {
    case all
    case singular
    case none
    
    var lapackJob : Int8 {
        switch self {
        case .all: return 0x41 /* "A" */
        case .singular: return 0x53 /* "S" */
        case .none: return 0x4E /* "N" */
        }
    }
}

public enum Transpose {
    case none
    case transpose
    case adjoint
    
    internal func blas(complex: Bool) -> Int8 {
        switch self {
        case .none: return   0x4E /* "N" */
        case .transpose: return 0x54 /* "T" */
        case .adjoint where complex: return 0x43 /* "C" */
        case .adjoint: return 0x54 /* "T" */
        }
    }
    
    internal var cblas : CBLAS_TRANSPOSE {
        switch self {
        case .none: return CblasNoTrans
        case .transpose: return CblasTrans
        case .adjoint: return CblasConjTrans
        }
    }
    
    public func apply<E>(_ matrix : Matrix<E>) -> Matrix<E> {
        switch self {
        case .none: return matrix
        case .transpose: return matrix.transpose
        case .adjoint: return matrix.adjoint
        }
    }
}

public protocol LANumeric : MatrixElement, AlgebraicField, ExpressibleByFloatLiteral where Magnitude : LANumeric  {
        
    init(magnitude : Self.Magnitude)

    var manhattanLength : Self.Magnitude { get }
    
    var length : Self.Magnitude { get }
    
    var lengthSquared : Self.Magnitude { get }
    
    var toInt : Int { get }
        
    static func random(in : ClosedRange<Self.Magnitude>) -> Self

    static func randomWhole(in : ClosedRange<Int>) -> Self
    
    static func blas_asum(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Self.Magnitude
    
    static func blas_nrm2(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Self.Magnitude
    
    static func blas_scal(_ N : Int32, _ alpha : Self, _ X : UnsafeMutablePointer<Self>, _ incX : Int32)
    
    static func blas_axpby(_ N : Int32,
                           _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                           _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    
    static func blas_iamax(_ N : Int32,
                           _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32
    
    static func blas_dot(_ N : Int32,
                         _ X : UnsafePointer<Self>, _ incX : Int32,
                         _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self
    
    static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self
    
    static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                          _ M : Int32, _ N : Int32, _ K : Int32,
                          _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                          _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)

    static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                          _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                          _ X : UnsafePointer<Self>, _ incX : Int32,
                          _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    
    static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                         _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                         _ Y : UnsafePointer<Self>, _ incY : Int32,
                         _ A : UnsafeMutablePointer<Self>, _ lda : Int32)

    static func blas_gerAdjoint(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    
    static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ ipiv : UnsafeMutablePointer<Int32>,
                            _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32

    static func lapack_gels(_ trans : Transpose,
                            _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                            _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32
    
    static func lapack_gesvd(_ jobu : UnsafeMutablePointer<Int8>, _ jobvt : UnsafeMutablePointer<Int8>,
                             _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>,
                             _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                             _ s : UnsafeMutablePointer<Self.Magnitude>,
                             _ u : UnsafeMutablePointer<Self>, _ ldu : UnsafeMutablePointer<Int32>,
                             _ vt : UnsafeMutablePointer<Self>, _ ldvt : UnsafeMutablePointer<Int32>,
                             _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                             _ info : UnsafeMutablePointer<Int32>) -> Int32

    static func lapack_heev(_ jobz : UnsafeMutablePointer<Int8>, _ uplo : UnsafeMutablePointer<Int8>, _ n : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ w : UnsafeMutablePointer<Self.Magnitude>,
                            _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32
    
    static func lapack_gees(_ jobvs : UnsafeMutablePointer<Int8>, _ n : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ wr : UnsafeMutablePointer<Self.Magnitude>,
                            _ wi : UnsafeMutablePointer<Self.Magnitude>,
                            _ vs : UnsafeMutablePointer<Self>, _ ldvs : UnsafeMutablePointer<Int32>,
                            _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32
    
    static func vDSP_elementwise_absolute(_ v : [Self]) -> [Self.Magnitude]
    
    static func vDSP_elementwise_adjoint(_ v : [Self]) -> [Self]
    
    static func vDSP_elementwise_multiply(_ u : [Self], _ v : [Self]) -> [Self]
    
    static func vDSP_elementwise_divide(_ u : [Self], _ v : [Self]) -> [Self]
    
}

public extension LANumeric {

    /// Computes the sum of the manhattanLengths of all elements in `vector`.
    static func manhattanLength(_ vector : Vector<Self>) -> Self.Magnitude {
         return blas_asum(Int32(vector.count), vector, 1)
    }

    /// Computes the L2 norm of this `vector`.
    static func length(_ vector : Vector<Self>) -> Self.Magnitude {
        return blas_nrm2(Int32(vector.count), vector, 1)
    }

    /// Scales vector `A` element-wise by `alpha` and stores the result in `A`.
    static func scaleVector(_ alpha : Self, _ A : inout Vector<Self>) {
        let C = Int32(A.count)
        asMutablePointer(&A) { A in
            blas_scal(C, alpha, A, 1)
        }
    }

    /// Scales vector `A` element-wise by `alpha`, scales matrix `B` element-wise by `beta`, and and stores the sum of the two scaled matrices in `B`.
    static func scaleAndAddVectors(_ alpha : Self, _ A : Vector<Self>, _ beta : Self, _ B : inout Vector<Self>) {
        precondition(A.count == B.count)
        asMutablePointer(&B) { B in
            blas_axpby(Int32(A.count), alpha, A, 1, beta, B, 1)
        }
    }
    
    /// Returns the index of the element with the largest manhattan length (-1 if the vector is empty).
    static func indexOfLargestElem(_ vector: Vector<Self>) -> Int {
        guard !vector.isEmpty else { return -1 }
        return Int(blas_iamax(Int32(vector.count), vector, 1))
    }
    
    /// Returns the dot product of `A` and `B`.
    static func dotProduct(_ A: Vector<Self>, _ B: Vector<Self>) -> Self {
        precondition(A.count == B.count)
        return blas_dot(Int32(A.count), A, 1, B, 1)
    }

    /// Returns the dot product of `A′` and `B`.
    static func adjointDotProduct(_ A: Vector<Self>, _ B: Vector<Self>) -> Self {
        precondition(A.count == B.count)
        return blas_adjointDot(Int32(A.count), A, 1, B, 1)
    }

    /// Scales the product of `A` and `B` by `alpha` and adds it to the result of scaling `C` by `beta`. Optionally `A` and / or `B` can be transposed/adjoint prior to that.
    static func matrixProduct(_ alpha : Self, _ transposeA : Transpose, _ A : Matrix<Self>, _ transposeB : Transpose, _ B : Matrix<Self>, _ beta : Self, _ C : inout Matrix<Self>) {
        let M = Int32(transposeA != .none ? A.columns : A.rows)
        let N = Int32(transposeB != .none ? B.rows : B.columns)
        let KA = Int32(transposeA != .none ? A.rows : A.columns)
        let KB = Int32(transposeB != .none ? B.columns : B.rows)
        precondition(KA == KB)
        precondition(M == C.rows)
        precondition(N == C.columns)
        guard M > 0 && N > 0 && KA > 0 else {
            scaleVector(beta, &C.elements)
            return
        }
        asMutablePointer(&C.elements) { C in
            let ta = transposeA.cblas
            let tb = transposeB.cblas
            blas_gemm(CblasColMajor, ta, tb, M, N, KA, alpha, A.elements, Int32(A.rows), B.elements, Int32(B.rows), beta, C, M)
        }
    }

    /// Scales the product of `A` and `X` by `alpha` and adds it to the result of scaling `Y` by `beta`. Optionally `A` can be transposed/adjoint prior to that.
    static func matrixVectorProduct(_ alpha : Self, _ transposeA : Transpose, _ A : Matrix<Self>, _ X : Vector<Self>, _ beta : Self, _ Y : inout Vector<Self>) {
        let M = Int32(A.rows)
        let N = Int32(A.columns)
        precondition(transposeA != .none ? (N == Y.count) : (N == X.count))
        precondition(transposeA != .none ? (M == X.count) : (M == Y.count))
        guard M > 0 && N > 0 else {
            scaleVector(beta, &Y)
            return
        }
        asMutablePointer(&Y) { Y in
            let ta = transposeA.cblas
            blas_gemv(CblasColMajor, ta, M, N, alpha, A.elements, M, X, 1, beta, Y, 1)
        }
    }

    /// Scales the product of `X` and  the transpose of `Y` by `alpha` and adds it to `A`.
    static func vectorVectorProduct(_ alpha : Self, _ X : Vector<Self>, _ Y : Vector<Self>, _ A : inout Matrix<Self>) {
        let M = Int32(A.rows)
        let N = Int32(A.columns)
        precondition(M == X.count)
        precondition(N == Y.count)
        guard M > 0 && N > 0 else { return }
        asMutablePointer(&A.elements) { A in
            blas_ger(CblasColMajor, M, N, alpha, X, 1, Y, 1, A, M)
        }
    }

    /// Scales the product of `X` and  the adjoint of `Y` by `alpha` and adds it to `A`.
    static func vectorAdjointVectorProduct(_ alpha : Self, _ X : Vector<Self>, _ Y : Vector<Self>, _ A : inout Matrix<Self>) {
        let M = Int32(A.rows)
        let N = Int32(A.columns)
        precondition(M == X.count)
        precondition(N == Y.count)
        guard M > 0 && N > 0 else { return }
        asMutablePointer(&A.elements) { A in
            blas_gerAdjoint(CblasColMajor, M, N, alpha, X, 1, Y, 1, A, M)
        }
    }
    
    /// Solves the system of linear equations `A * X = B` and stores the result `X` in `B`.
    /// - returns: `true` if the operation completed successfully, `false` otherwise.
    static func solveLinearEquations(_ A : Matrix<Self>, _ B : inout Matrix<Self>) -> Bool {
        var n = Int32(A.rows)
        precondition(A.columns == n && B.rows == n)
        var lda = n
        var ldb = n
        var info : Int32 = 0
        var nrhs = Int32(B.columns)
        var ipiv = [Int32](repeating: 0, count: Int(n))
        var A = A.elements
        asMutablePointer(&A) { A in
            asMutablePointer(&B.elements) { B in
                asMutablePointer(&ipiv) { ipiv in
                    let _ = lapack_gesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info)
                }
            }
        }
        return info == 0
    }
    
    /// Finds the minimum least squares solutions `x` of minimizing `(b - A * x).length` or `(b - A′ * x).length` and returns the result.
    /// Each column `x` in the result corresponds to the solution for the corresponding column `b` in `B`.
    static func solveLinearLeastSquares(_ A : Matrix<Self>, _ transposeA : Transpose, _ B : Matrix<Self>) -> Matrix<Self>? {
        var m : Int32 = Int32(A.rows)
        var n : Int32 = Int32(A.columns)
        let X = transposeA != .none ? A.rows : A.columns
        precondition(transposeA != .none ? B.rows == n : B.rows == m)
        var A = A
        A.extend(rows: 1)
        var B = B
        B.extend(rows: Int(max(1, max(n, m))))
        var nrhs : Int32 = Int32(B.columns)
        var lda : Int32 = Int32(A.rows)
        var ldb : Int32 = Int32(B.rows)
        var lwork : Int32 = -1
        var info : Int32 = 0
        asMutablePointer(&A.elements) { A in
            asMutablePointer(&B.elements) { B in
                var workCount : Self = 0
                let _ = lapack_gels(transposeA, &m, &n, &nrhs, A, &lda, B, &ldb, &workCount, &lwork, &info)
                guard info == 0 else { return }
                var work = [Self](repeating: 0, count: workCount.toInt)
                lwork = Int32(work.count)
                let _ = lapack_gels(transposeA, &m, &n, &nrhs, A, &lda, B, &ldb, &work, &lwork, &info)
            }
        }
        guard info == 0 else { return nil }
        return B[0 ..< X, 0 ..< B.columns]
    }

    /// Computes the singular value decomposition of a matrix`A` with `m` rows and `n` columns such that `A ≈ left * D * right`.
    /// Here `D == Matrix(rows: m, columns: n, diagonal: singularValues)` and `singularValues` has `min(m, n)` elements.
    /// The result matrix `left` has `m` rows, and depending on its job parameter either `m` (`all`), `min(m, n)` (`singular`) or `0` (`none`) columns.
    /// The result matrix `right` has `n` columns, and depending on its job parameter either `n` (`all`), `min(m, n)` (`singular`) or `0` (`none`) rows.
    static func singularValueDecomposition(_ A : Matrix<Self>, left : SVDJob, right : SVDJob) -> (singularValues : Vector<Self.Magnitude>, left : Matrix<Self>, right : Matrix<Self>)? {
        var m = Int32(A.rows)
        var n = Int32(A.columns)
        let k = min(m, n)
        let leftColumns : Int32
        switch left {
        case .all: leftColumns = m
        case .singular: leftColumns = k
        case .none: leftColumns = 0
        }
        let rightRows : Int32
        switch right {
        case .all: rightRows = n
        case .singular: rightRows = k
        case .none: rightRows = 0
        }
        guard k > 0 else {
            let U : Matrix<Self> = .eye(Int(m), Int(leftColumns))
            let VT : Matrix<Self> = .eye(Int(rightRows), Int(n))
            return (singularValues: [], left: U, right: VT)
        }
        var A = A
        var lda = m
        var S = Vector<Self.Magnitude>(repeating: 0, count: Int(k))
        var ldu = m
        var U = Matrix<Self>(rows: Int(ldu), columns: Int(leftColumns))
        var ldvt = max(1, rightRows)
        var VT = Matrix<Self>(rows: Int(ldvt), columns: Int(n))
        var jobleft = left.lapackJob
        var jobright = right.lapackJob
        var info : Int32 = 0
        var lwork : Int32 = -1
        asMutablePointer(&A.elements) { A in
            asMutablePointer(&S) { S in
                asMutablePointer(&U.elements) { U in
                    asMutablePointer(&VT.elements) { VT in
                        var workCount : Self = 0
                        let _ = lapack_gesvd(&jobleft, &jobright, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &workCount, &lwork, &info)
                        guard info == 0 else { return }
                        var work = [Self](repeating: 0, count: workCount.toInt)
                        lwork = Int32(work.count)
                        let _ = lapack_gesvd(&jobleft, &jobright, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &work, &lwork, &info)
                    }
                }
            }
        }
        if info == 0 {
            if rightRows == 0 {
                VT = Matrix<Self>(rows: 0, columns: VT.columns)
            }
            return (singularValues: S, left: U, right: VT)
        } else {
            return nil
        }
    }
    
    /// Computes the eigen decomposition of `A`. Here `A == A′` is assumed, and thus only the upper triangle part of `A` is used.
    /// The eigen vectors of `A` are stored in `A` if successful, otherwise the resulting content of `A` is undefined.
    /// - returns: `nil` if the decomposition failed, otherwise the eigenvalues of `A`
    static func eigenDecomposition(_ A : inout Matrix<Self>) -> Vector<Magnitude>? {
        var n = Int32(A.rows)
        precondition(n == A.columns)
        guard n > 0 else { return [] }
        var jobz : Int8 = 0x56 /* "V" */
        var uplo : Int8 = 0x55 /* "U" */
        var lda = n
        var w : [Magnitude] = Array(repeating: 0, count: Int(n))
        var info : Int32 = 0
        var lwork : Int32 = -1
        asMutablePointer(&A.elements) { a in
            asMutablePointer(&w) { w in
                var workCount : Self = 0
                let _ = lapack_heev(&jobz, &uplo, &n, a, &lda, w, &workCount, &lwork, &info)
                guard info == 0 else { return }
                var work = [Self](repeating: 0, count: workCount.toInt)
                lwork = Int32(work.count)
                let _ = lapack_heev(&jobz, &uplo, &n, a, &lda, w, &work, &lwork, &info)
            }
        }
        if info == 0 {
            return w
        } else {
            return nil
        }
    }
    
    /// Computes schur decomposition of `A`. The schur form is
    static func schurDecomposition<R>(_ A : inout Matrix<Self>) -> (eigenValues : Vector<Complex<R>>, schurVectors : Matrix<Self>)? where R == Self.Magnitude {
        var n = Int32(A.rows)
        precondition(n == A.columns)
        guard n > 0 else { return (eigenValues : [], schurVectors : Matrix()) }
        var jobvs : Int8 = 0x56 /* "V" */
        var lda = n
        var w : [Complex<R>] = Array(repeating: 0, count: Int(n))
        var wr : [Self.Magnitude] = Array(repeating: 0, count: Int(n))
        var wi : [Self.Magnitude] = Array(repeating: 0, count: Int(n))
        var info : Int32 = 0
        var lwork : Int32 = -1
        var vs = Matrix<Self>(rows: Int(n), columns: Int(n))
        var ldvs = n
        asMutablePointer(&A.elements) { a in
            asMutablePointer(&w) { w in
                asMutablePointer(&vs.elements) { vs in
                    var workCount : Self = 0
                    let _ = lapack_gees(&jobvs, &n, a, &lda, &wr, &wi, vs, &ldvs, &workCount, &lwork, &info)
                    guard info == 0 else { return }
                    var work = [Self](repeating: 0, count: workCount.toInt)
                    lwork = Int32(work.count)
                    let _ = lapack_gees(&jobvs, &n, a, &lda, &wr, &wi, vs, &ldvs, &work, &lwork, &info)
                }
            }
        }
        if info == 0 {
            for i in 0 ..< A.rows {
                w[i] = Complex(wr[i], wi[i])
            }
            return (eigenValues: w, schurVectors : vs)
        } else {
            return nil
        }
    }

}

infix operator ′* : MultiplicationPrecedence
infix operator *′ : MultiplicationPrecedence
infix operator ′*′ : MultiplicationPrecedence
infix operator ∖ : MultiplicationPrecedence // unicode character "set minus": U+2216
infix operator ′∖ : MultiplicationPrecedence // unicode character "set minus": U+2216

infix operator .* : MultiplicationPrecedence
infix operator ./ : MultiplicationPrecedence
infix operator .- : AdditionPrecedence
infix operator .+ : AdditionPrecedence

public extension Matrix where Element : Numeric {
    
    static func eye(_ m : Int) -> Matrix {
        return Matrix(diagonal : [Element](repeating: 1, count: m))
    }
    
    static func eye(_ m : Int, _ n : Int) -> Matrix {
        return Matrix(rows: m, columns: n, diagonal : [Element](repeating: 1, count: min(n, m)))
    }

}

public extension Matrix where Element : LANumeric {
    
    var manhattanNorm : Element.Magnitude { return Element.manhattanLength(elements) }
    
    var norm : Element.Magnitude { return Element.length(elements) }
    
    var largest : Element {
        let index = Element.indexOfLargestElem(elements)
        if index >= 0 {
            return elements[index]
        } else {
            return 0
        }
    }
    
    var maxNorm : Element.Magnitude {
        return largest.manhattanLength
    }
        
    static func + (left : Matrix, right : Matrix) -> Matrix {
        var result = left
        result += right
        return result
    }
    
    static func - (left : Matrix, right : Matrix) -> Matrix {
        var result = left
        result -= right
        return result
    }
    
    static func += (left : inout Matrix, right : Matrix) {
        left.accumulate(1, 1, right)
    }
    
    static func -= (left : inout Matrix, right : Matrix) {
        left.accumulate(1, -1, right)
    }
        
    mutating func accumulate(_ alpha : Element, _ beta : Element, _ other : Matrix<Element>) {
        precondition(hasSameDimensions(other))
        Element.scaleAndAddVectors(beta, other.elements, alpha, &self.elements)
    }
            
    static func * (left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.rows, columns: right.columns)
        Element.matrixProduct(1, .none, left, .none, right, 0, &C)
        return C
    }
    
    static func ′* (left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.columns, columns: right.columns)
        Element.matrixProduct(1, .adjoint, left, .none, right, 0, &C)
        return C
    }

    static func *′ (left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.rows, columns: right.rows)
        Element.matrixProduct(1, .none, left, .adjoint, right, 0, &C)
        return C
    }

    static func ′*′ (left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.columns, columns: right.rows)
        Element.matrixProduct(1, .adjoint, left, .adjoint, right, 0, &C)
        return C
    }

    static func * (left : Matrix, right : Vector<Element>) -> Vector<Element> {
        var Y = [Element](repeating: Element.zero, count: left.rows)
        Element.matrixVectorProduct(1, .none, left, right, 0, &Y)
        return Y
    }
    
    static func ′* (left : Matrix, right : Vector<Element>) -> Vector<Element> {
        var Y = [Element](repeating: Element.zero, count: left.columns)
        Element.matrixVectorProduct(1, .adjoint, left, right, 0, &Y)
        return Y
    }

    static func *= (left : inout Matrix, right : Element) {
        Element.scaleVector(right, &left.elements)
    }
    
    static func * (left : Element, right : Matrix) -> Matrix {
        var A = right
        A *= left
        return A
    }
    
    static func .+(left : Matrix, right : Matrix) -> Matrix {
        return left + right
    }

    static func .-(left : Matrix, right : Matrix) -> Matrix {
        return left - right
    }

    static func .*(left : Matrix, right : Matrix) -> Matrix {
        precondition(left.hasSameDimensions(right))
        let elems = Element.vDSP_elementwise_multiply(left.elements, right.elements)
        return Matrix(rows: left.rows, columns: left.columns, elements: elems)
    }

    static func ./(left : Matrix, right : Matrix) -> Matrix {
        precondition(left.hasSameDimensions(right))
        let elems = Element.vDSP_elementwise_divide(left.elements, right.elements)
        return Matrix(rows: left.rows, columns: left.columns, elements: elems)
    }

    func solve(_ rhs : Matrix) -> Matrix? {
        var B = rhs
        if Element.solveLinearEquations(self, &B) {
            return B
        } else {
            return nil
        }
    }
    
    func solve(_ rhs : Vector<Element>) -> Vector<Element>? {
        return solve(Matrix(rhs))?.vector
    }
    
    func solveLeastSquares(transpose : Transpose = .none, _ rhs : Matrix) -> Matrix? {
        return Element.solveLinearLeastSquares(self, transpose, rhs)
    }
    
    func solveLeastSquares(transpose : Transpose = .none, _ rhs : Vector<Element>) -> Vector<Element>? {
        return Element.solveLinearLeastSquares(self, transpose, Matrix(rhs))?.vector
    }
    
    func svd(left : SVDJob = .all, right : SVDJob = .all) -> (singularValues : Vector<Element.Magnitude>, left : Matrix, right : Matrix)? {
        return Element.singularValueDecomposition(self, left: left, right: right)
    }
    
    func eigen() -> (eigenValues : Vector<Element.Magnitude>, eigenVectors : Matrix<Element>)? {
        var A = self
        guard let eigenValues = Element.eigenDecomposition(&A) else { return nil }
        return (eigenValues: eigenValues, eigenVectors: A)
    }
    
    func schur<R>() -> (eigenValues : Vector<Complex<R>>, schurForm : Matrix<Element>, schurVectors : Matrix<Element>)? where R == Element.Magnitude {
        var A = self
        guard let result = Element.schurDecomposition(&A) else { return nil }
        return (eigenValues: result.eigenValues, schurForm: A, schurVectors: result.schurVectors)
    }

    static func ∖ (lhs : Matrix, rhs : Matrix) -> Matrix {
        return lhs.solveLeastSquares(rhs)!
    }
    
    static func ∖ (lhs : Matrix, rhs : Vector<Element>) -> Vector<Element> {
        return lhs.solveLeastSquares(rhs)!
    }

    static func ′∖ (lhs : Matrix, rhs : Matrix) -> Matrix {
        return lhs.solveLeastSquares(transpose: .adjoint, rhs)!
    }
    
    static func ′∖ (lhs : Matrix, rhs : Vector<Element>) -> Vector<Element> {
        return lhs.solveLeastSquares(transpose: .adjoint, rhs)!
    }

}

public func * <Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Element {
    return Element.dotProduct(left, right)
}

public func ′* <Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Element {
    return Element.adjointDotProduct(left, right)
}

public func *′ <Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Matrix<Element> {
    var A = Matrix<Element>(rows: left.count, columns: right.count)
    Element.vectorAdjointVectorProduct(1, left, right, &A)
    return A
}

public postfix func ′<Element : LANumeric>(vector : Vector<Element>) -> Matrix<Element> {
    return Matrix(row: Element.vDSP_elementwise_adjoint(vector))
}

public func .+<Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Vector<Element> {
    var B = right
    Element.scaleAndAddVectors(1, left, 1, &B)
    return B
}

public func .-<Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Vector<Element> {
    var B = right
    Element.scaleAndAddVectors(1, left, -1, &B)
    return B
}

public func .*<Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Vector<Element> {
    return Element.vDSP_elementwise_multiply(left, right)
}

public func ./<Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Vector<Element> {
    return Element.vDSP_elementwise_divide(left, right)
}




