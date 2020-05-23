# LANumerics

Copyright (c) 2020 Steven Obua

License: MIT License

*LANumerics* is a Swift package for doing *numerical linear algebra*. 

The package depends on [Swift Numerics](https://github.com/apple/swift-numerics), as it supports both **real** and **complex** numerics for both `Float` and `Double` precision in a uniform way. 
Under the hood it relies on the [`Accelerate`](https://developer.apple.com/documentation/accelerate) framework for most of its functionality, in particular `BLAS` and `LAPACK`, and also `vDSP`.

## Usage
*LANumerics* is a normal Swift package and can be added to your App [in the usual way](https://developer.apple.com/documentation/xcode/adding_package_dependencies_to_your_app).
After adding it to your app, import `LANumerics` (and also `Numerics` if you use complex numbers). 

You can try out if everything works fine by running

```
import LANumerics
let A : Matrix<Float> = Matrix(columns: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("A: \(A)")
```

which should output something like

```
A: 4x3-matrix:
⎛1.0  5.0  9.0 ⎞
⎜2.0  6.0  10.0⎟
⎜3.0  7.0  11.0⎟
⎝4.0  8.0  12.0⎠
```

## LANumeric

The `LANumeric` protocol denotes the type of numbers on which *LANumerics* operates. It is implemented by the following types:

*  `Float`
*  `Double`
*  `Complex<Float>`
*  `Complex<Double>`

Most functionality of *LANumerics* is generic in `LANumeric`, e.g. solving a system of linear equations or computing the singular value decomposition of a matrix.

## Constructing Matrices

The main work horse of *LANumerics* is the `Matrix` type. For convenience there is also a `Vector` type, but this is just a typealias for normal Swift arrays.

The expression `Matrix([1,2,3])` constructs the matrix:
```
3x1-matrix:
⎛1.0⎞
⎜2.0⎟
⎝3.0⎠
```
The expression `Matrix(row: [1, 2, 3])` constructs the matrix:
```
1x3-matrix:
(1.0  2.0  3.0)
```
The expression `Matrix<Float>(rows: 2, columns: 3)` constructs a matrix consisting only of zeros:
```
2x3-matrix:
⎛0.0  0.0  0.0⎞
⎝0.0  0.0  0.0⎠
```
The expression `Matrix(repeating: 1, rows: 2, columns: 3)` constructs a matrix consisting only of ones:
```
2x3-matrix:
⎛1.0  1.0  1.0⎞
⎝1.0  1.0  1.0⎠
```
Given the two vectors `v1` and `v2`
```
let v1 : Vector<Float> = [1, 2, 3]
let v2 : Vector<Float> = [4, 5, 6]
```
we can create a matrix from columns `Matrix(columns: [v1, v2])`:
```
3x2-matrix:
⎛1.0  4.0⎞
⎜2.0  5.0⎟
⎝3.0  6.0⎠
```
or rows `Matrix(rows: [v1, v2])`:
```
2x3-matrix:
⎛1.0  2.0  3.0⎞
⎝4.0  5.0  6.0⎠
```
It is also legal to create matrices with zero columns and/or rows, like `Matrix(rows: 2, columns: 0)` or `Matrix(rows: 0, columns: 0)`.

## SIMD Support

Swift supports `simd` vector and matrix operations. *LANumerics* plays nice with `simd` by providing conversion functions to and from `simd` vectors and matrices. For example,
```
import simd
import LANumerics

let m = Matrix(rows: [[1, 2, 3], [4, 5, 6]])
print("m: \(m)")
let s = m.simd3x2
print("------------")
print("as simd: \(s)")
print("------------")
print(Matrix(s) == m)
```
results in the output 
```
m: 2x3-matrix:
⎛1.0  2.0  3.0⎞
⎝4.0  5.0  6.0⎠
------------
as simd: simd_double3x2(columns: (SIMD2<Double>(1.0, 4.0), SIMD2<Double>(2.0, 5.0), SIMD2<Double>(3.0, 6.0)))
------------
true
```
Note that `simd` reverses the role of row and column indices compared to `LANumerics` (and usual mathematical convention).

## Accessing Matrix Elements and Submatrices

## Matrix Arithmetic

*LANumerics* supports common operations on matrices, among them:

* `transpose` and `adjoint`
* matrix multiplication
* elementwise operations

### Transpose and Adjoint

### Matrix Multiplication

### Elementwise Operations


## Solving Linear Equations

## Matrix Decompositions

## More

Complete documentation of the *LANumerics* API will eventually become available, but this is, just like the package itself, still work in progress. 
If you feel like experimenting and exploring more of the currently available functionality, examining the [current tests](https://github.com/phlegmaticprogrammer/LANumerics/tree/master/Tests/LANumericsTests) should provide a good starting point.

