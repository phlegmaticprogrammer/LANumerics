import Accelerate

public extension LANumericPrimitives {

    /// Computes the sum of the manhattanLengths of all elements in `vector`.
    static func manhattanLength(_ vector : Vector<Self>) -> Self.Magnitude {
        return blas_asum(Int32(vector.count), vector, 1)
    }

    /// Computes the L2 norm of this `vector`.
    static func length(_ vector : Vector<Self>) -> Self.Magnitude {
        return blas_nrm2(Int32(vector.count), vector, 1)
    }

    /// Scales matrix `A` element-wise by `alpha` and stores the result in `A`.
    static func scaleVector(_ alpha : Self, _ A : inout Vector<Self>) {
        let C = Int32(A.count)
        asMutablePointer(&A) { A in
            blas_scal(C, alpha, A, 1)
        }
    }

    /// Scales matrix `A` element-wise by `alpha`, scales matrix `B` element-wise by `beta`, and and stores the sum of the two scaled matrices in `B`.
    static func scaleAndAddVectors(_ alpha : Self, _ A : Vector<Self>, _ beta : Self, _ B : inout Vector<Self>) {
        precondition(A.count == B.count)
        asMutablePointer(&B) { B in
            blas_axpby(Int32(A.count), alpha, A, 1, beta, B, 1)
        }
    }
    
    /// Returns the index of the element with the largest length (-1 if the vector is empty).
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
    static func adjointDotProduct(_ A: Vector<Self>, _ B: Vector<Self>) -> Self.Magnitude {
        precondition(A.count == B.count)
        return blas_adjointDot(Int32(A.count), A, 1, B, 1)
    }

    /// Scales the product of `A` and `B` by `alpha` and adds it to the result of scaling `C` by `beta`. Optionally `A` and / or `B` can be transposed prior to that.
    static func matrixProduct(_ alpha : Self, _ transposeA : Bool, _ A : Matrix<Self>, _ transposeB : Bool, _ B : Matrix<Self>, _ beta : Self, _ C : inout Matrix<Self>) {
        let M = Int32(transposeA ? A.columns : A.rows)
        let N = Int32(transposeB ? B.rows : B.columns)
        let KA = Int32(transposeA ? A.rows : A.columns)
        let KB = Int32(transposeB ? B.columns : B.rows)
        precondition(KA == KB)
        precondition(M == C.rows)
        precondition(N == C.columns)
        guard M > 0 && N > 0 && KA > 0 else {
            scaleVector(beta, &C.elements)
            return
        }
        asMutablePointer(&C.elements) { C in
            let ta = transposeA ? CblasTrans : CblasNoTrans
            let tb = transposeB ? CblasTrans : CblasNoTrans
            blas_gemm(CblasColMajor, ta, tb, M, N, KA, alpha, A.elements, Int32(A.rows), B.elements, Int32(B.rows), beta, C, M)
        }
    }

    /// Scales the product of `A` and `X` by `alpha` and adds it to the result of scaling `Y` by `beta`. Optionally `A` can be transposed prior to that.
    static func matrixVectorProduct(_ alpha : Self, _ transposeA : Bool, _ A : Matrix<Self>, _ X : Vector<Self>, _ beta : Self, _ Y : inout Vector<Self>) {
        let M = Int32(A.rows)
        let N = Int32(A.columns)
        precondition(transposeA ? (N == Y.count) : (N == X.count))
        precondition(transposeA ? (M == X.count) : (M == Y.count))
        guard M > 0 else {
            scaleVector(beta, &Y)
            return
        }
        asMutablePointer(&Y) { Y in
            let ta = transposeA ? CblasTrans : CblasNoTrans
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
    
    /// Finds the minimum least squares solutions `x` of minimizing `(b - A * x).euclideanNorm` or `(b - A′ * x).euclideanNorm` and returns the result.
    /// Each column `x` in the result corresponds to the solution for the corresponding column `b` in `B`.
    static func solveLinearLeastSquares(_ A : Matrix<Self>, _ transposeA : Bool, _ B : Matrix<Self>) -> Matrix<Self>? {
        var trans : Int8 = (transposeA ? 0x54 /* "T" */ : 0x4E /* "N" */)
        var m : Int32 = Int32(A.rows)
        var n : Int32 = Int32(A.columns)
        let X = transposeA ? A.rows : A.columns
        precondition(transposeA ? B.rows == n : B.rows == m)
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
                let _ = lapack_gels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &workCount, &lwork, &info)
                guard info == 0 else { return }
                var work = [Self](repeating: 0, count: workCount.toInt)
                lwork = Int32(work.count)
                let _ = lapack_gels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &work, &lwork, &info)
            }
        }
        guard info == 0 else { return nil }
        return B[0 ..< X, 0 ..< B.columns]
    }

    /// Computes the singular value decomposition of a matrix`A` with `m` rows and `n` columns such that `A ≈ left * D * right`.
    /// Here `D == Matrix(rows: m, columns: n, diagonal: singularValues)` and `singularValues` has `min(m, n)` elements.
    /// The result matrix `left` has `m` rows, and depending on its job parameter either `m` (`all`), `min(m, n)` (`singular`) or `0` (`none`) columns.
    /// The result matrix `right` has `n` columns, and depending on its job parameter either `n` (`all`), `min(m, n)` (`singular`) or `0` (`none`) rows.
    static func singularValueDecomposition(_ A : Matrix<Self>, left : SVDJob, right : SVDJob) -> (singularValues : Vector<Self>, left : Matrix<Self>, right : Matrix<Self>)? {
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
        var S = Vector<Self>(repeating: 0, count: Int(k))
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


}




