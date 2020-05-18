postfix operator ′ // unicode character "Prime": 2082

public postfix func ′<Element : MatrixElement>(vector : Vector<Element>) -> Matrix<Element> {
    return Matrix(row: vector)
}

public postfix func ′<Element : MatrixElement>(matrix : Matrix<Element>) -> Matrix<Element> {
    return matrix.transposed()
}

public typealias BlockMatrix<Element : MatrixElement> = Matrix<Matrix<Element>>

public extension Matrix {
    
    mutating func transpose() {
        if _rows > 1 && _columns > 1 {
            var transposedElements = elements
            asPointer(elements) { elements in
                asMutablePointer(&transposedElements) { transposedElements in
                    for r in 0 ..< _rows {
                        for c in 0 ..< _columns {
                            let sourceIndex = c * _rows + r
                            let targetIndex = r * _columns + c
                            transposedElements[targetIndex] = elements[sourceIndex]
                        }
                    }
                }
            }
            elements = transposedElements
        }
        swap(&_rows, &_columns)
    }
    
    func transposed() -> Matrix {
        var m = self
        m.transpose()
        return m
    }
        
    func column(_ c : Int) -> Matrix {
        precondition(c >= 0 && c < _columns)
        let start = c * _rows
        return Matrix(Array(elements[start ..< start + _rows]))
    }
    
    func row(_ r : Int) -> Matrix {
        precondition(r >= 0 && r < _rows)
        let elems = (0 ..< _columns).map { c in self[r, c] }
        return Matrix(row: elems)
    }
    
    mutating func reshape(rows : Int, columns : Int) {
        precondition(rows * columns == count)
        self._rows = rows
        self._columns = columns
    }
    
    func reshaped(rows : Int, columns : Int) -> Matrix {
        var A = self
        A.reshape(rows: rows, columns: columns)
        return A
    }

    /// - todo: This is a naive implementation, needs to be optimized.
    subscript <R : Collection, C : Collection>(rowIndices : R, columnIndices : C) -> Matrix where R.Element == Int, C.Element == Int {
        get {
            var elems : [Element] = []
            for c in columnIndices {
                for r in rowIndices {
                    elems.append(self[r, c])
                }
            }
            return Matrix(rows: rowIndices.count, columns: columnIndices.count, elements: elems)
        }
        set {
            precondition(newValue.rows == rowIndices.count && newValue.columns == columnIndices.count)
            var index = 0
            let elems = newValue.elements
            for c in columnIndices {
                for r in rowIndices {
                    self[r, c] = elems[index]
                    index += 1
                }
            }
        }
    }
    
    init(columns : [Vector<Element>]) {
        var matrix = BlockMatrix<Element>(rows : 1, columns : columns.count)
        for (i, column) in columns.enumerated() {
            matrix[0, i] = Matrix(column)
        }
        self = flatten(matrix)
    }

    init(rows : [Vector<Element>]) {
        var matrix = BlockMatrix<Element>(rows : rows.count, columns : 1)
        for (i, row) in rows.enumerated() {
            matrix[i, 0] = Matrix(row: row)
        }
        self = flatten(matrix)
    }

    init(stackHorizontally stack: [Matrix<Element>]) {
        var matrix = BlockMatrix<Element>(rows : 1, columns : stack.count)
        for (i, m) in stack.enumerated() {
            matrix[0, i] = m
        }
        self = flatten(matrix)
    }

    init(stackVertically stack: [Matrix<Element>]) {
        var matrix = BlockMatrix<Element>(rows : stack.count, columns : 1)
        for (i, m) in stack.enumerated() {
            matrix[i, 0] = m
        }
        self = flatten(matrix)
    }
}

public func flatten<E : MatrixElement>(_ matrix : BlockMatrix<E>) -> Matrix<E> {
    var rowHeights = [Int](repeating: 0, count: matrix.rows)
    var columnWidths = [Int](repeating: 0, count: matrix.columns)
    var totalRows = 0
    var totalColumns = 0
    for r in 0 ..< matrix.rows {
        var height = 0
        for c in 0 ..< matrix.columns {
            height = max(height, matrix[r, c].rows)
        }
        rowHeights[r] = height
        totalRows += height
    }
    for c in 0 ..< matrix.columns {
        var width = 0
        for r in 0 ..< matrix.rows {
            width = max(width, matrix[r, c].columns)
        }
        columnWidths[c] = width
        totalColumns += width
    }
    var result = Matrix<E>(rows: totalRows, columns: totalColumns)
    var targetColumn = 0
    for c in 0 ..< matrix.columns {
        var targetRow = 0
        for r in 0 ..< matrix.rows {
            let m = matrix[r, c]
            result[targetRow ..< targetRow + m.rows, targetColumn ..< targetColumn + m.columns] = m
            targetRow += rowHeights[r]
        }
        targetColumn += columnWidths[c]
    }
    return result
}

