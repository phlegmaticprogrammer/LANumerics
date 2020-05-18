import Foundation

public extension Matrix {
    
    func map<E>(_ transform : (Element) throws -> E) rethrows -> Matrix<E> {
        return Matrix<E>(rows: _rows, columns: _columns, elements: try elements.map(transform))
    }

    func combine<E, F>(_ other : Matrix<E>, _ using : (Element, E) throws -> F) rethrows -> Matrix<F> {
        precondition(hasSameDimensions(other))
        let C = count
        var elems = [F](repeating: F.zero, count: C)
        try asPointer(self.elements) { elems1 in
            try asPointer(other.elements) { elems2 in
                try asMutablePointer(&elems) { elems in
                    for i in 0 ..< C {
                        elems[i] = try using(elems1[i], elems2[i])
                    }
                }
            }
        }
        return Matrix<F>(rows: _rows, columns: _columns, elements: elems)
    }
    
    func fold<F>(_ start : F, _ using : (F, Element) throws -> F) rethrows -> F {
        var result = start
        for elem in elements {
            result = try using(result, elem)
        }
        return result
    }
    
    func fold(_ using : (Element, Element) throws -> Element) rethrows -> Element {
        return try fold(Element.zero, using)
    }
    
    func forall(_ cond : (Element) throws -> Bool) rethrows -> Bool {
        for elem in elements {
            guard try cond(elem) else { return false }
        }
        return true
    }

    func exists(_ cond : (Element) throws -> Bool) rethrows -> Bool {
        for elem in elements {
            if try cond(elem) { return true }
        }
        return false
    }

    
}
