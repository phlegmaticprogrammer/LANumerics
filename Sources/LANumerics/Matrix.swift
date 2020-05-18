public protocol MatrixElement : Hashable {

    static var zero : Self { get }
    
}

public typealias Vector<Elem : MatrixElement> = [Elem]

public struct Matrix<Elem : MatrixElement> : MatrixElement {
    
    /// We keep elements in [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
    var elements : Vector<Elem>
    
    var _rows : Int
    
    var _columns : Int
    
    /// The number of rows of the matrix.
    public var rows : Int {
        return _rows
    }
    
    /// The number of columns of the matrix.
    public var columns : Int {
        return _columns
    }
        
    public init(repeating : Elem = Elem.zero, rows : Int, columns : Int) {
        precondition(rows >= 0 && columns >= 0)
        self._rows = rows
        self._columns = columns
        elements = [Elem](repeating: repeating, count: rows * columns)
    }
    
    public init(rows : Int, columns : Int, elements : (_ row : Int, _ column : Int) -> Elem) {
        precondition(rows >= 0 && columns >= 0)
        self._rows = rows
        self._columns = columns
        self.elements = [Elem](repeating: Elem.zero, count: rows * columns)
        var index = 0
        for c in 0 ..< columns {
            for r in 0 ..< rows {
                self.elements[index] = elements(r, c)
                index += 1
            }
        }
    }
    
    public init(rows : Int, columns : Int, elements : Vector<Elem>) {
        precondition(rows >= 0 && columns >= 0 && rows * columns == elements.count)
        self._rows = rows
        self._columns = columns
        self.elements = elements
    }
    
    public init(row : Vector<Elem>) {
        self._rows = 1
        self._columns = row.count
        self.elements = row
    }
    
    public init(_ column : Vector<Elem>) {
        self._columns = 1
        self._rows = column.count
        self.elements = column
    }
    
    public init(_ singleton : Elem) {
        self._columns = 1
        self._rows = 1
        self.elements = [singleton]
    }
        
    public init(diagonal : Vector<Elem>) {
        let m = diagonal.count
        self._rows = m
        self._columns = m
        self.elements = [Elem](repeating: Elem.zero, count: m * m)
        for i in 0 ..< columns {
            self[i, i] = diagonal[i]
        }
    }
    
    public func hasDimensions(_ rows : Int, _ columns : Int) -> Bool {
        return rows == self._rows && columns == self._columns
    }
    
    public static func zeros(_ rows : Int, _ columns : Int) -> Matrix {
        return Matrix(rows: rows, columns: columns)
    }

    private func indexIsValid(row : Int, column : Int) -> Bool {
        return row >= 0 && column >= 0 && row < _rows && column < _columns
    }
    
    func hasSameDimensions<F>(_ other : Matrix<F>) -> Bool {
        return _rows == other._rows && _columns == other._columns
    }
    
    public var isRowVector : Bool { return _rows == 1 }
    
    public var isColumnVector : Bool { return _columns == 1 }

    public var isVector : Bool { return _rows == 1 || _columns == 1 }
    
    /// Returns the number of elements in this matrix.
    public var count : Int { return _rows * _columns }
    
    /// Returns the matrix as vector. This also succeeds if the matrix is not an actual vector, the elements are then in column-major order.
    public var vector : Vector<Elem> { return elements }

    public subscript(row : Int, column : Int) -> Elem {
        get {
            precondition(indexIsValid(row: row, column: column))
            return elements[column * _rows + row]
        }
        set {
            precondition(indexIsValid(row: row, column: column))
            elements[column * _rows + row] = newValue
        }
    }

    /// Gets / sets the element at `index` when viewing the matrix as a vector (with elements in column-major order).
    public subscript(index : Int) -> Elem {
        get {
            return elements[index]
        }
        set {
            elements[index] = newValue
        }
    }
    
    public static var zero: Matrix<Elem> {
        return Matrix<Elem>(rows: 0, columns: 0)
    }
    
}

