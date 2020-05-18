func asMutablePointer<T, R>(_ array : inout [T], block : (UnsafeMutablePointer<T>) throws -> R) rethrows -> R {
    let count = array.count
    return try array.withUnsafeMutableBytes { ptr in
        try block(ptr.baseAddress!.bindMemory(to: T.self, capacity: count))
    }
}

func asPointer<T, R>(_ array : [T], block : (UnsafePointer<T>) throws -> R) rethrows -> R {
    let count = array.count
    return try array.withUnsafeBytes { ptr in
        try block(ptr.baseAddress!.bindMemory(to: T.self, capacity: count))
    }
}

