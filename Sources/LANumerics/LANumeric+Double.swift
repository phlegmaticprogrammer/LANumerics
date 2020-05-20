import Accelerate

extension Double : LANumeric {
        
    public var manhattanLength : Double { return magnitude }

    public var adjoint : Double { return self }

    public var length : Self.Magnitude { return magnitude }
    
    public var lengthSquared : Self.Magnitude { return magnitude * magnitude }

    public var toInt : Int {
        return Int(self)
    }

    public init(magnitude: Self.Magnitude) {
        self = magnitude
    }

    public static func randomWhole(in range : ClosedRange<Int>) -> Self {
        return Double(Int.random(in: range))
    }

    public static func blas_asum(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_dasum(N, X, incX)
    }
    
    public static func blas_nrm2(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_dnrm2(N, X, incX)
    }

    public static func blas_scal(_ N : Int32, _ alpha : Self, _ X : UnsafeMutablePointer<Self>, _ incX : Int32) {
        cblas_dscal(N, alpha, X, incX)
    }

    public static func blas_axpby(_ N : Int32, _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32, _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32) {
        catlas_daxpby(N, alpha, X, incX, beta, Y, incY)
    }
    
    public static func blas_iamax(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32 {
        cblas_idamax(N, X, incX)
    }
    
    public static func blas_dot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        cblas_ddot(N, X, incX, Y, incY)
    }
    
    public static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self.Magnitude {
        cblas_ddot(N, X, incX, Y, incY)
    }

    public static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                                 _ M : Int32, _ N : Int32, _ K : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                                 _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)
    {
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    }

    public static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                                 _ X : UnsafePointer<Self>, _ incX : Int32,
                                 _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    {
        cblas_dgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
    }

    public static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    {
        cblas_dger(Order, M, N, alpha, X, incX, Y, incY, A, lda)
    }

    public static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ ipiv : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        dgesv_(n, nrhs, a, lda, ipiv, b, ldb, info)
    }
    
    public static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                                   _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        dgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
    }

    public static func lapack_gesvd(_ jobu : UnsafeMutablePointer<Int8>, _ jobvt : UnsafeMutablePointer<Int8>,
                                    _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>,
                                    _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                    _ s : UnsafeMutablePointer<Self>,
                                    _ u : UnsafeMutablePointer<Self>, _ ldu : UnsafeMutablePointer<Int32>,
                                    _ vt : UnsafeMutablePointer<Self>, _ ldvt : UnsafeMutablePointer<Int32>,
                                    _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                    _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
    }

}
