public protocol LANumeric : MatrixElement, Numeric {

    /// Computes the sum of the absolute values of all elements in `vector`.
    static func manhattanNorm(_ vector : Vector<Self>) -> Self.Magnitude

    /// Computes the euclidean norm of this `vector`.
    static func euclideanNorm(_ vector : Vector<Self>) -> Self.Magnitude
    
    /// Scales matrix `A` element-wise by `alpha` and stores the result in `A`.
    static func scaleVector(_ alpha : Self, _ A : inout Vector<Self>)

    /// Scales matrix `A` element-wise by `alpha`, scales matrix `B` element-wise by `beta`, and and stores the sum of the two scaled matrices in `B`.
    static func scaleAndAddVectors(_ alpha : Self, _ A : Vector<Self>, _ beta : Self, _ B : inout Vector<Self>)

    /// Returns the index of the element with the largest absolute value (-1 if the vector is empty).
    static func indexOfLargestElem(_ vector : Vector<Self>) -> Int
    
    /// Returns the dot product of `A` and `B`.
    static func dotProduct(_ A : Vector<Self>, _ B : Vector<Self>) -> Self
    
    /// Scales the product of `A` and `B` by `alpha` and adds it to the result of scaling `C` by `beta`. Optionally `A` and / or `B` can be transposed prior to that.
    static func matrixProduct(_ alpha : Self, _ transposeA : Bool, _ A : Matrix<Self>, _ transposeB : Bool, _ B : Matrix<Self>, _ beta : Self, _ C : inout Matrix<Self>)
    
    /// Scales the product of `A` and `X` by `alpha` and adds it to the result of scaling `Y` by `beta`. Optionally `A` can be transposed prior to that.
    static func matrixVectorProduct(_ alpha : Self, _ transposeA : Bool, _ A : Matrix<Self>, _ X : Vector<Self>, _ beta : Self, _ Y : inout Vector<Self>)

    /// Scales the product of `X` and  the transpose of `Y` by `alpha` and adds it to `A`.
    static func vectorVectorProduct(_ alpha : Self, _ X : Vector<Self>, _ Y : Vector<Self>, _ A : inout Matrix<Self>)
    
    /// Solves the system of linear equations `A * X = B` and stores the result `X` in `B`. 
    /// - returns: `true` if the operation completed successfully, `false` otherwise.
    static func solveLinearEquations(_ A : Matrix<Self>, _ B : inout Matrix<Self>) -> Bool

}

infix operator ′* : MultiplicationPrecedence
infix operator *′ : MultiplicationPrecedence
infix operator ′*′ : MultiplicationPrecedence

public extension Matrix where Elem : LANumeric {
    
    var manhattanNorm : Elem.Magnitude { return Elem.manhattanNorm(elements) }
    
    var euclideanNorm : Elem.Magnitude { return Elem.euclideanNorm(elements) }
    
    var largest : Elem {
        let index = Elem.indexOfLargestElem(elements)
        if index >= 0 {
            return elements[index]
        } else {
            return 0
        }
    }
        
    static func +(left : Matrix, right : Matrix) -> Matrix {
        return left.scaleAndAdd(1, 1, right)
    }
    
    static func -(left : Matrix, right : Matrix) -> Matrix {
        return left.scaleAndAdd(1, -1, right)
    }
    
    func scaleAndAdd(_ alpha : Elem, _ beta : Elem, _ other : Matrix<Elem>) -> Matrix<Elem> {
        var result = self
        result.accumulate(alpha, beta, other)
        return result
    }
    
    mutating func accumulate(_ alpha : Elem, _ beta : Elem, _ other : Matrix<Elem>) {
        precondition(hasSameDimensions(other))
        Elem.scaleAndAddVectors(beta, other.elements, alpha, &self.elements)
    }
        
    static func eye(_ m : Int) -> Matrix {
        return Matrix(diagonal : [Elem](repeating: 1, count: m))
    }
    
    static func *(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Elem>(rows: left.rows, columns: right.columns)
        Elem.matrixProduct(1, false, left, false, right, 0, &C)
        return C
    }
    
    static func ′*(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Elem>(rows: left.columns, columns: right.columns)
        Elem.matrixProduct(1, true, left, false, right, 0, &C)
        return C
    }

    static func *′(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Elem>(rows: left.rows, columns: right.rows)
        Elem.matrixProduct(1, false, left, true, right, 0, &C)
        return C
    }

    static func ′*′(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Elem>(rows: left.columns, columns: right.rows)
        Elem.matrixProduct(1, true, left, true, right, 0, &C)
        return C
    }

    static func *(left : Matrix, right : Vector<Elem>) -> Vector<Elem> {
        var Y = [Elem](repeating: Elem.zero, count: left.rows)
        Elem.matrixVectorProduct(1, false, left, right, 0, &Y)
        return Y
    }
    
    static func ′*(left : Matrix, right : Vector<Elem>) -> Vector<Elem> {
        var Y = [Elem](repeating: Elem.zero, count: left.columns)
        Elem.matrixVectorProduct(1, true, left, right, 0, &Y)
        return Y
    }

    static func *=(left : inout Matrix, right : Elem) {
        Elem.scaleVector(right, &left.elements)
    }
    
    static func *(left : Elem, right : Matrix) -> Matrix {
        var A = right
        A *= left
        return A
    }
    
}

public extension Matrix where Elem : LANumeric {
    
    var infinityNorm : Elem.Magnitude { return largest.magnitude }
    
}

public func ′* <Elem : LANumeric>(left : Vector<Elem>, right : Vector<Elem>) -> Elem {
    return Elem.dotProduct(left, right)
}

public func *′ <Elem : LANumeric>(left : Vector<Elem>, right : Vector<Elem>) -> Matrix<Elem> {
    var A = Matrix<Elem>(rows: left.count, columns: right.count)
    Elem.vectorVectorProduct(1, left, right, &A)
    return A
}

