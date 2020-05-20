import Accelerate
import Numerics

extension Complex : LANumeric, ExpressibleByFloatLiteral {
        
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
        var _alpha = alpha
        dispatch(
            float: { cblas_caxpy(N, &_alpha, X, incX, Y, incY) },
            double: { cblas_zaxpy(N, &_alpha, X, incX, Y, incY) }
        )
    }
    
    public static func blas_iamax(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Int32 {
        dispatch(
            float: { cblas_icamax(N, X, incX) },
            double: { cblas_izamax(N, X, incX) }
        )
    }

    public static func blas_dot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }
    
    public static func blas_adjointDot(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32, _ Y : UnsafePointer<Self>, _ incY : Int32) -> Self {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }

    public static func blas_gemm(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ TransB : CBLAS_TRANSPOSE,
                                 _ M : Int32, _ N : Int32, _ K : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32, _ B : UnsafePointer<Self>, _ ldb : Int32,
                                 _ beta : Self, _ C : UnsafeMutablePointer<Self>, _ ldc : Int32)
    {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }

    public static func blas_gemv(_ Order : CBLAS_ORDER, _ TransA : CBLAS_TRANSPOSE, _ M : Int32, _ N : Int32,
                                 _ alpha : Self, _ A : UnsafePointer<Self>, _ lda : Int32,
                                 _ X : UnsafePointer<Self>, _ incX : Int32,
                                 _ beta : Self, _ Y : UnsafeMutablePointer<Self>, _ incY : Int32)
    {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }

    public static func blas_ger(_ Order : CBLAS_ORDER, _ M : Int32, _ N : Int32,
                                _ alpha : Self, _ X : UnsafePointer<Self>, _ incX : Int32,
                                _ Y : UnsafePointer<Self>, _ incY : Int32,
                                _ A : UnsafeMutablePointer<Self>, _ lda : Int32)
    {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }

    public static func lapack_gesv(_ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ ipiv : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }

    public static func lapack_gels(_ trans : UnsafeMutablePointer<Int8>,
                                   _ m : UnsafeMutablePointer<Int32>, _ n : UnsafeMutablePointer<Int32>, _ nrhs : UnsafeMutablePointer<Int32>,
                                   _ a : UnsafeMutablePointer<Self>, _ lda : UnsafeMutablePointer<Int32>,
                                   _ b : UnsafeMutablePointer<Self>, _ ldb : UnsafeMutablePointer<Int32>,
                                   _ work : UnsafeMutablePointer<Self>, _ lwork : UnsafeMutablePointer<Int32>,
                                   _ info : UnsafeMutablePointer<Int32>) -> Int32
    {
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
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
        dispatch(
            float: { fatalError() },
            double: { fatalError() }
        )
    }
}

