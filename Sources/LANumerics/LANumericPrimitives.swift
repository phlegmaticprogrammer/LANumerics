import Foundation
import Numerics
import Accelerate

public protocol LANumericPrimitives : MatrixElement, Numeric, ExpressibleByFloatLiteral {
        
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
    
    static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self.Magnitude
    
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
    
    static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ ipiv : UnsafeMutablePointer<Int32>,
                            _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32

    static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                            _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                            _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                            _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                            _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                            _ info : UnsafeMutablePointer<Int32>) -> Int32
    
    static func lapack_gesvd(_ jobu : UnsafeMutablePointer<Int8>, _ jobvt : UnsafeMutablePointer<Int8>,
                             _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>,
                             _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                             _ s : UnsafeMutablePointer<Self>,
                             _ u : UnsafeMutablePointer<Self>, _ ldu : UnsafeMutablePointer<Int32>,
                             _ vt : UnsafeMutablePointer<Self>, _ ldvt : UnsafeMutablePointer<Int32>,
                             _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                             _ info : UnsafeMutablePointer<Int32>) -> Int32

}

extension Float : LANumericPrimitives {
    
    public var manhattanLength : Float { return magnitude }
    
    public var adjoint : Float { return self }
    
    public var length : Self.Magnitude { return magnitude }
    
    public var lengthSquared : Self.Magnitude { return magnitude * magnitude }
    
    public var toInt : Int {
        return Int(self)
    }

    public init(magnitude: Self.Magnitude) {
        self = magnitude
    }

    public static func randomWhole(in range : ClosedRange<Int>) -> Self {
        return Float(Int.random(in: range))
    }

    public static func blas_asum(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_sasum(N, X, incX)
    }
    
    public static func blas_nrm2(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_snrm2(N, X, incX)
    }

    public static func blas_scal(_ N : Int32, _ alpha : Self, _ X : UnsafeMutablePointer<Self>, _ incX : Int32) {
        cblas_sscal(N, alpha, X, incX)
    }

    public static func blas_axpby(_ N : Int32, _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32, _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32) {
        fatalError()
    }
    
    public static func blas_iamax(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32 {
        fatalError()
    }
    
    public static func blas_dot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        fatalError()
    }
    
    public static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self.Magnitude {
        fatalError()
    }

    public static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                                 _ M : Int32, _ N : Int32, _ K : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                                 _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)
    {
        fatalError()
    }

    public static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                                 _ X : UnsafePointer<Self>, _ incX : Int32,
                                 _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    {
        fatalError()
    }

    public static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    {
        fatalError()
    }

    public static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ ipiv : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
    }
    
    public static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                                   _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
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
        fatalError()
    }

}

extension Double : LANumericPrimitives {
        
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
        fatalError()
    }

    public static func blas_iamax(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32 {
        fatalError()
    }
    
    public static func blas_dot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        fatalError()
    }
    
    public static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self.Magnitude {
        fatalError()
    }

    public static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                                 _ M : Int32, _ N : Int32, _ K : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                                 _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)
    {
        fatalError()
    }

    public static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                                 _ X : UnsafePointer<Self>, _ incX : Int32,
                                 _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    {
        fatalError()
    }

    public static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    {
        fatalError()
    }

    public static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ ipiv : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
    }

    public static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                                   _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
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
        fatalError()
    }

}

extension Complex : LANumericPrimitives, ExpressibleByFloatLiteral {
        
    public typealias FloatLiteralType = Double
    
    static func dispatch<R>(float : () -> R, double : () -> R) -> R {
        if RealType.self == Float.self {
            return float()
        } else if RealType.self == Double.self {
            return double()
        } else {
            fatalError("cannot dispatch on Complex.RealType == \(RealType.self)")
        }
    }
    
    public init(floatLiteral: Self.FloatLiteralType) {
        var x : RealType = 0
        Complex.dispatch(
            float: { x = Float(floatLiteral) as! RealType },
            double: { x = Double(floatLiteral) as! RealType }
        )
        self.init(x)
    }
    
    public var manhattanLength : Magnitude { return real.magnitude + imaginary.magnitude }

    public var adjoint : Complex { return conjugate }

    public init(magnitude: Self.Magnitude) {
        self = Complex(magnitude, 0)
    }

    public var toInt : Int {
        precondition(imaginary.isZero)
        return Complex.dispatch (
            float: { return Int(real as! Float) },
            double: { return Int(real as! Double) }
        )
    }

    public static func random(in range: ClosedRange<RealType>) -> Self {
        if RealType.self == Float.self {
            let r = range as! ClosedRange<Float>
            let x = Float.random(in: r) as! RealType
            let y = Float.random(in: r) as! RealType
            return Complex(x, y)
        } else if RealType.self == Double.self {
            let r = range as! ClosedRange<Double>
            let x = Double.random(in: r) as! RealType
            let y = Double.random(in: r) as! RealType
            return Complex(x, y)
        } else {
            fatalError()
        }
    }
    
    public static func randomWhole(in range : ClosedRange<Int>) -> Self {
        if RealType.self == Float.self {
            let x = Float.randomWhole(in: range) as! RealType
            let y = Float.randomWhole(in: range) as! RealType
            return Complex(x, y)
        } else if RealType.self == Double.self {
            let x = Double.randomWhole(in: range) as! RealType
            let y = Double.randomWhole(in: range) as! RealType
            return Complex(x, y)
        } else {
            fatalError()
        }
    }

    public static func blas_asum(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        if RealType.self == Float.self {
            return cblas_scasum(N, X, incX) as! Self.Magnitude
        } else if RealType.self == Double.self {
            return cblas_dzasum(N, X, incX) as! Self.Magnitude
        } else {
            fatalError()
        }
    }
    
    public static func blas_nrm2(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        if RealType.self == Float.self {
            return cblas_scnrm2(N, X, incX) as! Self.Magnitude
        } else if RealType.self == Double.self {
            return cblas_dznrm2(N, X, incX) as! Self.Magnitude
        } else {
            fatalError()
        }
    }
    
    public static func blas_scal(_ N : Int32, _ alpha : Self, _ X : UnsafeMutablePointer<Self>, _ incX : Int32) {
        var _alpha = alpha
        dispatch(
            float: { cblas_cscal(N, &_alpha, X, incX) },
            double: { cblas_zscal(N, &_alpha, X, incX) }
        )
    }

    public static func blas_axpby(_ N : Int32, _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32, _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32) {
        fatalError()
    }
    
    public static func blas_iamax(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32 {
        fatalError()
    }

    public static func blas_dot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        fatalError()
    }
    
    public static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self.Magnitude {
        fatalError()
    }

    public static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                                 _ M : Int32, _ N : Int32, _ K : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                                 _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)
    {
        fatalError()
    }

    public static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                                 _ X : UnsafePointer<Self>, _ incX : Int32,
                                 _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    {
        fatalError()
    }

    public static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    {
        fatalError()
    }

    public static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ ipiv : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
    }

    public static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                                   _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        fatalError()
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
        fatalError()
    }
}

public extension LANumericPrimitives {

    static func manhattanLength(_ vector : Vector<Self>) -> Self.Magnitude {
        return blas_asum(Int32(vector.count), vector, 1)
    }

    static func length(_ vector : Vector<Self>) -> Self.Magnitude {
        return blas_nrm2(Int32(vector.count), vector, 1)
    }


    static func scaleVector(_ alpha : Self, _ A : inout Vector<Self>) {
        let C = Int32(A.count)
        asMutablePointer(&A) { A in
            blas_scal(C, alpha, A, 1)
        }
    }

    static func scaleAndAddVectors(_ alpha : Self, _ A : Vector<Self>, _ beta : Self, _ B : inout Vector<Self>) {
        precondition(A.count == B.count)
        asMutablePointer(&B) { B in
            blas_axpby(Int32(A.count), alpha, A, 1, beta, B, 1)
        }
    }
    
    static func indexOfLargestElem(_ vector: Vector<Self>) -> Int {
        guard !vector.isEmpty else { return -1 }
        return Int(blas_iamax(Int32(vector.count), vector, 1))
    }
    
    static func dotProduct(_ A: Vector<Self>, _ B: Vector<Self>) -> Self {
        precondition(A.count == B.count)
        return blas_dot(Int32(A.count), A, 1, B, 1)
    }

    static func adjointDotProduct(_ A: Vector<Self>, _ B: Vector<Self>) -> Self.Magnitude {
        precondition(A.count == B.count)
        return blas_adjointDot(Int32(A.count), A, 1, B, 1)
    }

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



