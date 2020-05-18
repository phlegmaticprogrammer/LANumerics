import simd

public extension Matrix where Element : SIMDScalar {

    init(_ simd : SIMD2<Element>) {
        self.init([simd.x, simd.y])
    }

    init(_ simd : SIMD3<Element>) {
        self.init([simd.x, simd.y, simd.z])
    }
    
    init(_ simd : SIMD4<Element>) {
        self.init([simd.x, simd.y, simd.z, simd.w])
    }
    
    init(_ simd : SIMD8<Element>) {
        self.init((0 ..< 8).map { i in simd[i] })
    }

    init(_ simd : SIMD16<Element>) {
        self.init((0 ..< 16).map { i in simd[i] })
    }

    init(_ simd : SIMD32<Element>) {
        self.init((0 ..< 32).map { i in simd[i] })
    }

    init(_ simd : SIMD64<Element>) {
        self.init((0 ..< 64).map { i in simd[i] })
    }

    init(row simd : SIMD2<Element>) {
        self.init(row: [simd.x, simd.y])
    }

    init(row simd : SIMD3<Element>) {
        self.init(row: [simd.x, simd.y, simd.z])
    }
    
    init(row simd : SIMD4<Element>) {
        self.init(row: [simd.x, simd.y, simd.z, simd.w])
    }
    
    init(row simd : SIMD8<Element>) {
        self.init(row: (0 ..< 8).map { i in simd[i] })
    }

    init(row simd : SIMD16<Element>) {
        self.init(row: (0 ..< 16).map { i in simd[i] })
    }

    init(row simd : SIMD32<Element>) {
        self.init(row: (0 ..< 32).map { i in simd[i] })
    }

    init(row simd : SIMD64<Element>) {
        self.init(row: (0 ..< 64).map { i in simd[i] })
    }
    
    var simd2 : SIMD2<Element> {
        return SIMD2(elements)
    }
    
    var simd3 : SIMD3<Element> {
        return SIMD3(elements)
    }
    
    var simd4 : SIMD4<Element> {
        return SIMD4(elements)
    }

    var simd8 : SIMD8<Element> {
        return SIMD8(elements)
    }

    var simd16 : SIMD16<Element> {
        return SIMD16(elements)
    }

    var simd32 : SIMD32<Element> {
        return SIMD32(elements)
    }

    var simd64 : SIMD64<Element> {
        return SIMD64(elements)
    }

}

public extension Matrix where Element == Float {
    
    init(_ simd : simd_float2x2) {
        let (col0, col1) = simd.columns
        self.init(rows: 2, columns: 2, elements: [col0[0], col0[1], col1[0], col1[1]])
    }
    
    init(_ simd : simd_float2x3) {
        let (col0, col1) = simd.columns
        self.init(rows: 3, columns: 2, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2]])
    }
    
    init(_ simd : simd_float2x4) {
        let (col0, col1) = simd.columns
        self.init(rows: 4, columns: 2, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3]])
    }
    
    init(_ simd : simd_float3x2) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 2, columns: 3, elements: [col0[0], col0[1], col1[0], col1[1], col2[0], col2[1]])
    }

    init(_ simd : simd_float3x3) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 3, columns: 3, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]])
    }

    init(_ simd : simd_float3x4) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 4, columns: 3, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3]])
    }

    init(_ simd : simd_float4x2) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 2, columns: 4, elements: [col0[0], col0[1], col1[0], col1[1], col2[0], col2[1], col3[0], col3[1]])
    }

    init(_ simd : simd_float4x3) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 3, columns: 4, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2], col2[0], col2[1], col2[2], col3[0], col3[1], col3[2]])
    }

    init(_ simd : simd_float4x4) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 4, columns: 4, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3], col3[0], col3[1], col3[2], col3[3]])
    }

    var simd2x2 : simd_float2x2 {
        precondition(hasDimensions(2, 2))
        return simd_float2x2(columns: (column(0).simd2, column(1).simd2))
    }
    
    var simd2x3 : simd_float2x3 {
        precondition(hasDimensions(3, 2))
        return simd_float2x3(columns: (column(0).simd3, column(1).simd3))
    }
    
    var simd2x4 : simd_float2x4 {
        precondition(hasDimensions(4, 2))
        return simd_float2x4(columns: (column(0).simd4, column(1).simd4))
    }
    
    var simd3x2 : simd_float3x2 {
        precondition(hasDimensions(2, 3))
        return simd_float3x2(columns: (column(0).simd2, column(1).simd2, column(2).simd2))
    }

    var simd3x3 : simd_float3x3 {
        precondition(hasDimensions(3, 3))
        return simd_float3x3(columns: (column(0).simd3, column(1).simd3, column(2).simd3))
    }

    var simd3x4 : simd_float3x4 {
        precondition(hasDimensions(4, 3))
        return simd_float3x4(columns: (column(0).simd4, column(1).simd4, column(2).simd4))
    }

    var simd4x2 : simd_float4x2 {
        precondition(hasDimensions(2, 4))
        return simd_float4x2(columns: (column(0).simd2, column(1).simd2, column(2).simd2, column(3).simd2))
    }

    var simd4x3 : simd_float4x3 {
        precondition(hasDimensions(3, 4))
        return simd_float4x3(columns: (column(0).simd3, column(1).simd3, column(2).simd3, column(3).simd3))
    }

    var simd4x4 : simd_float4x4 {
        precondition(hasDimensions(4, 4))
        return simd_float4x4(columns: (column(0).simd4, column(1).simd4, column(2).simd4, column(3).simd4))
    }
    
}

public extension Matrix where Element == Double {
    
    init(_ simd : simd_double2x2) {
        let (col0, col1) = simd.columns
        self.init(rows: 2, columns: 2, elements: [col0[0], col0[1], col1[0], col1[1]])
    }
    
    init(_ simd : simd_double2x3) {
        let (col0, col1) = simd.columns
        self.init(rows: 3, columns: 2, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2]])
    }
    
    init(_ simd : simd_double2x4) {
        let (col0, col1) = simd.columns
        self.init(rows: 4, columns: 2, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3]])
    }
    
    init(_ simd : simd_double3x2) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 2, columns: 3, elements: [col0[0], col0[1], col1[0], col1[1], col2[0], col2[1]])
    }

    init(_ simd : simd_double3x3) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 3, columns: 3, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]])
    }

    init(_ simd : simd_double3x4) {
        let (col0, col1, col2) = simd.columns
        self.init(rows: 4, columns: 3, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3]])
    }

    init(_ simd : simd_double4x2) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 2, columns: 4, elements: [col0[0], col0[1], col1[0], col1[1], col2[0], col2[1], col3[0], col3[1]])
    }

    init(_ simd : simd_double4x3) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 3, columns: 4, elements: [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2], col2[0], col2[1], col2[2], col3[0], col3[1], col3[2]])
    }

    init(_ simd : simd_double4x4) {
        let (col0, col1, col2, col3) = simd.columns
        self.init(rows: 4, columns: 4, elements: [col0[0], col0[1], col0[2], col0[3], col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3], col3[0], col3[1], col3[2], col3[3]])
    }

    var simd2x2 : simd_double2x2 {
        precondition(hasDimensions(2, 2))
        return simd_double2x2(columns: (column(0).simd2, column(1).simd2))
    }
    
    var simd2x3 : simd_double2x3 {
        precondition(hasDimensions(3, 2))
        return simd_double2x3(columns: (column(0).simd3, column(1).simd3))
    }
    
    var simd2x4 : simd_double2x4 {
        precondition(hasDimensions(4, 2))
        return simd_double2x4(columns: (column(0).simd4, column(1).simd4))
    }
    
    var simd3x2 : simd_double3x2 {
        precondition(hasDimensions(2, 3))
        return simd_double3x2(columns: (column(0).simd2, column(1).simd2, column(2).simd2))
    }

    var simd3x3 : simd_double3x3 {
        precondition(hasDimensions(3, 3))
        return simd_double3x3(columns: (column(0).simd3, column(1).simd3, column(2).simd3))
    }

    var simd3x4 : simd_double3x4 {
        precondition(hasDimensions(4, 3))
        return simd_double3x4(columns: (column(0).simd4, column(1).simd4, column(2).simd4))
    }

    var simd4x2 : simd_double4x2 {
        precondition(hasDimensions(2, 4))
        return simd_double4x2(columns: (column(0).simd2, column(1).simd2, column(2).simd2, column(3).simd2))
    }

    var simd4x3 : simd_double4x3 {
        precondition(hasDimensions(3, 4))
        return simd_double4x3(columns: (column(0).simd3, column(1).simd3, column(2).simd3, column(3).simd3))
    }

    var simd4x4 : simd_double4x4 {
        precondition(hasDimensions(4, 4))
        return simd_double4x4(columns: (column(0).simd4, column(1).simd4, column(2).simd4, column(3).simd4))
    }
    
}

