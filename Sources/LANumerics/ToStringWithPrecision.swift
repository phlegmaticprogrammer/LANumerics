import Foundation

public protocol ToStringWithPrecision {
        
    func toString(precision : Int?) -> String

}

extension Float : ToStringWithPrecision {
        
    public func toString(precision : Int?) -> String {
        if let precision = precision {
            return String(format: "%.\(precision)f", self)
        } else {
            return self.description
        }
    }
    
}

extension Double : ToStringWithPrecision {
        
    public func toString(precision : Int?) -> String {
        if let precision = precision {
            return String(format: "%.\(precision)f", self)
        } else {
            return self.description
        }
    }
    
}

extension Matrix : ToStringWithPrecision, CustomStringConvertible where Element : ToStringWithPrecision {
    
    private func left(row : Int) -> String {
        guard rows > 1 else { return "(" }
        if row == 0 { return "⎛" }
        if row == rows - 1 { return "⎝" }
        return "⎜"
    }
    
    private func right(row : Int) -> String {
        guard rows > 1 else { return ")" }
        if row == 0 { return "⎞" }
        if row == rows - 1 { return "⎠" }
        return "⎟"
    }
    
    private func extend(_ s : String, width : Int) -> String {
        var t = s
        while t.count < width {
            t.append(" ")
        }
        return t
    }

    public func toString(precision : Int? = nil) -> String {
        var column_widths = [Int](repeating: 0, count: columns)
        for c in 0 ..< columns {
            var W = 0
            for r in 0 ..< rows {
                let width = self[r, c].toString(precision: precision).count
                W = max(width, W)
            }
            column_widths[c] = W
        }
        var s : String = ""
        for r in 0 ..< rows {
            if r > 0 { s.append("\n") }
            s.append(left(row: r))
            for c in 0 ..< columns {
                if c > 0 { s.append("  ") }
                s.append(extend(self[r, c].toString(precision: precision), width: column_widths[c]))
            }
            s.append(right(row: r))
        }
        return s
    }
    
    public var description: String {
        if _rows == 0 || _columns == 0 {
            return "\(_rows)x\(_columns)-matrix"
        } else {
            return "\(_rows)x\(_columns)-matrix:\n\(toString(precision: nil))"
        }
    }

}
