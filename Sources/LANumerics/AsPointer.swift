func asMutablePointer<T, R>(_ array : inout [T], block : (UnsafeMutablePointer<T>) -> R) -> R {
    let count = array.count
    return array.withUnsafeMutableBytes { ptr in
        block(ptr.baseAddress!.bindMemory(to: T.self, capacity: count))
    }
}

func asPointer<T, R>(_ array : [T], block : (UnsafePointer<T>) -> R) -> R {
    let count = array.count
    return array.withUnsafeBytes { ptr in
        block(ptr.baseAddress!.bindMemory(to: T.self, capacity: count))
    }
}

