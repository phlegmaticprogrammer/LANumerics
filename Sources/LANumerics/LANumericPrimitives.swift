import Foundation
import Numerics
import Accelerate

public protocol LANumericPrimitives : MatrixElement, Numeric {
    
    static func random(in : ClosedRange<Self.Magnitude>) -> Self

    static func randomWhole(in : ClosedRange<Int>) -> Self
    
    static func blas_asum(_ N : Int32, _ X : UnsafePointer<Self>, _ incX : Int32) -> Self.Magnitude
    
}

extension Float : LANumericPrimitives {

    public static func randomWhole(in range : ClosedRange<Int>) -> Self {
        return Float(Int.random(in: range))
    }

    public static func blas_asum(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_sasum(N, X, incX)
    }

}

extension Double : LANumericPrimitives {
        
    public static func randomWhole(in range : ClosedRange<Int>) -> Self {
        return Double(Int.random(in: range))
    }

    public static func blas_asum(_ N: Int32, _ X: UnsafePointer<Self>, _ incX: Int32) -> Self.Magnitude {
        return cblas_dasum(N, X, incX)
    }

}

/*fileprivate func recast<U, V>(_ X : UnsafePointer<U>) -> UnsafePointer<V> {
    let _X = UnsafeRawPointer(X)
    return _X.assumingMemoryBound(to: V.self)
}*/

extension Complex : LANumericPrimitives {
    
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
    
}

