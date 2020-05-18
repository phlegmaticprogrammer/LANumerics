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
    
    /// Finds the minimum least squares solutions `x` of minimizing `(b - A * x).euclideanNorm` or `(b - A′ * x).euclideanNorm` and returns the result.
    /// Each column `x` in the result corresponds to the solution for the corresponding column `b` in `B`.
    static func solveLinearLeastSquares(_ A : Matrix<Self>, _ transposeA : Bool, _ B : Matrix<Self>) -> Matrix<Self>?

}

infix operator ′* : MultiplicationPrecedence
infix operator *′ : MultiplicationPrecedence
infix operator ′*′ : MultiplicationPrecedence
infix operator ∖ : MultiplicationPrecedence // unicode character "set minus": U+2216
infix operator ′∖ : MultiplicationPrecedence // unicode character "set minus": U+2216

public extension Matrix where Element : LANumeric {
    
    var manhattanNorm : Element.Magnitude { return Element.manhattanNorm(elements) }
    
    var euclideanNorm : Element.Magnitude { return Element.euclideanNorm(elements) }
    
    var largest : Element {
        let index = Element.indexOfLargestElem(elements)
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
    
    func scaleAndAdd(_ alpha : Element, _ beta : Element, _ other : Matrix<Element>) -> Matrix<Element> {
        var result = self
        result.accumulate(alpha, beta, other)
        return result
    }
    
    mutating func accumulate(_ alpha : Element, _ beta : Element, _ other : Matrix<Element>) {
        precondition(hasSameDimensions(other))
        Element.scaleAndAddVectors(beta, other.elements, alpha, &self.elements)
    }
        
    static func eye(_ m : Int) -> Matrix {
        return Matrix(diagonal : [Element](repeating: 1, count: m))
    }
    
    static func *(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.rows, columns: right.columns)
        Element.matrixProduct(1, false, left, false, right, 0, &C)
        return C
    }
    
    static func ′*(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.columns, columns: right.columns)
        Element.matrixProduct(1, true, left, false, right, 0, &C)
        return C
    }

    static func *′(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.rows, columns: right.rows)
        Element.matrixProduct(1, false, left, true, right, 0, &C)
        return C
    }

    static func ′*′(left : Matrix, right : Matrix) -> Matrix {
        var C = Matrix<Element>(rows: left.columns, columns: right.rows)
        Element.matrixProduct(1, true, left, true, right, 0, &C)
        return C
    }

    static func *(left : Matrix, right : Vector<Element>) -> Vector<Element> {
        var Y = [Element](repeating: Element.zero, count: left.rows)
        Element.matrixVectorProduct(1, false, left, right, 0, &Y)
        return Y
    }
    
    static func ′*(left : Matrix, right : Vector<Element>) -> Vector<Element> {
        var Y = [Element](repeating: Element.zero, count: left.columns)
        Element.matrixVectorProduct(1, true, left, right, 0, &Y)
        return Y
    }

    static func *=(left : inout Matrix, right : Element) {
        Element.scaleVector(right, &left.elements)
    }
    
    static func *(left : Element, right : Matrix) -> Matrix {
        var A = right
        A *= left
        return A
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
    
    func solveLeastSquares(transpose : Bool = false, _ rhs : Matrix) -> Matrix? {
        return Element.solveLinearLeastSquares(self, transpose, rhs)
    }
    
    func solveLeastSquares(transpose : Bool = false, _ rhs : Vector<Element>) -> Vector<Element>? {
        return Element.solveLinearLeastSquares(self, transpose, Matrix(rhs))?.vector
    }
    
    static func ∖(lhs : Matrix, rhs : Matrix) -> Matrix {
        return lhs.solveLeastSquares(rhs)!
    }
    
    static func ∖(lhs : Matrix, rhs : Vector<Element>) -> Vector<Element> {
        return lhs.solveLeastSquares(rhs)!
    }

    static func ′∖(lhs : Matrix, rhs : Matrix) -> Matrix {
        return lhs.solveLeastSquares(transpose: true, rhs)!
    }
    
    static func ′∖(lhs : Matrix, rhs : Vector<Element>) -> Vector<Element> {
        return lhs.solveLeastSquares(transpose: true, rhs)!
    }
}

public extension Matrix where Element : LANumeric {
    
    var infinityNorm : Element.Magnitude { return largest.magnitude }
    
}

public func ′* <Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Element {
    return Element.dotProduct(left, right)
}

public func *′ <Element : LANumeric>(left : Vector<Element>, right : Vector<Element>) -> Matrix<Element> {
    var A = Matrix<Element>(rows: left.count, columns: right.count)
    Element.vectorVectorProduct(1, left, right, &A)
    return A
}

