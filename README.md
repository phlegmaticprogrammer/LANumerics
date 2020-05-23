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

```swift
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
```swift
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
```swift
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

Matrix elements and submatrices can be accessed using familiar notation:
```swift
import simd
import LANumerics

var m = Matrix(rows: [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m)
m[2, 1] = 0
print(m)
print(m[0 ... 1, 0 ... 1])
print(m[1 ... 2, 1 ... 2])
m[0 ... 1, 0 ... 1] = m[1 ... 2, 1 ... 2]
print(m)
```
yields the output
```
3x3-matrix:
⎛1.0  2.0  3.0⎞
⎜4.0  5.0  6.0⎟
⎝7.0  8.0  9.0⎠
3x3-matrix:
⎛1.0  2.0  3.0⎞
⎜4.0  5.0  6.0⎟
⎝7.0  0.0  9.0⎠
2x2-matrix:
⎛1.0  2.0⎞
⎝4.0  5.0⎠
2x2-matrix:
⎛5.0  6.0⎞
⎝0.0  9.0⎠
3x3-matrix:
⎛5.0  6.0  3.0⎞
⎜0.0  9.0  6.0⎟
⎝7.0  0.0  9.0⎠
```

## Matrix Arithmetic

*LANumerics* supports common operations on matrices and vectors, among them:

* `transpose` and `adjoint`
* matrix multiplication
* vector products 

In the following, assume the context
```swift
import Numerics
import LANumerics
let u = Matrix<Complex<Float>>(rows: [[1, 2 * .i], [3, 4 * .i + 1]])
let v = Matrix<Complex<Float>>(rows: [[.i, 0], [0, 1 + 1 * .i]])
print("u : \(u)\n")
print("v : \(v)\n")
```
which has output 
```
u : 2x2-matrix:
⎛1.0  2.0i      ⎞
⎝3.0  1.0 + 4.0i⎠

v : 2x2-matrix:
⎛1.0i  0.0       ⎞
⎝0.0   1.0 + 1.0i⎠
```

### Transpose and Adjoint
For **real** matrices, `transpose` and `adjoint` have the same meaning, but for **complex** matrices the `adjoint` is the element-wise conjugate of the `transpose`. Executing
```swift
print("u.transpose : \(u.transpose)\n")
print("u.adjoint : \(u.adjoint)\n")
```
thus yields
```
u.transpose : 2x2-matrix:
⎛1.0   3.0       ⎞
⎝2.0i  1.0 + 4.0i⎠

u.adjoint : 2x2-matrix:
⎛1.0    3.0       ⎞
⎝-2.0i  1.0 - 4.0i⎠
```
The `adjoint` has the advantage over the `transpose` that many properties involving the adjoint generalize naturally from real matrices to complex matrices. Therefore there is the shortcut notation
```swift
u′
```
for `u.adjoint`. 

Note that `′` is the unicode character "Prime" `U+2032`. You can use for example [Ukelele](https://software.sil.org/ukelele/) to make the input of that character smooth. Other alternatives are configuring the touchbar of your macbook, or
using a configurable keyboard like [Stream Deck](https://www.elgato.com/en/gaming/stream-deck).

### Matrix Multiplication

Multiplying `u` and `v` is done via the expression `u * v`. Running `print("u * v: \(u * v)")` results in
```
u * v: 2x2-matrix:
⎛1.0i  -2.0 + 2.0i⎞
⎝3.0i  -3.0 + 5.0i⎠
```

Instead of `u′ * v` one can also use the equivalent, but usually faster expression `u ′* v`:
```
print("u′ * v: \(u′ * v)\n")
print("u ′* v: \(u ′* v)\n")
```
yields
```
u′ * v: 2x2-matrix:
⎛1.0i  3.0 + 3.0i⎞
⎝2.0   5.0 - 3.0i⎠

u ′* v: 2x2-matrix:
⎛1.0i  3.0 + 3.0i⎞
⎝2.0   5.0 - 3.0i⎠
```
Similarly, it is better to use `u *′ v` than `u * v′`, and `u ′*′ v` instead of `u′ * v′`.

### Vector Products

We will view `u` and `v` as vectors `u.vector` and `v.vector` now, where `.vector` corresponds to a [*column-major*](https://en.wikipedia.org/wiki/Row-_and_column-major_order) order of the matrix elements:
```
u.vector: [Complex<Float>(1.0, 0.0), Complex<Float>(3.0, 0.0), Complex<Float>(0.0, 2.0), Complex<Float>(1.0, 4.0)]
v.vector: [Complex<Float>(0.0, 1.0), Complex<Float>(0.0, 0.0), Complex<Float>(0.0, 0.0), Complex<Float>(1.0, 1.0)]
```

The dot product of `u.vector` and `v.vector`  results in 
```
u.vector * v.vector: -3.0 + 6.0i
```
Another vector product is 
```
u.vector ′* v.vector: 5.0 - 2.0i
```
which corresponds to 
```
u.vector′ * v.vector: [Complex<Float>(5.0, -2.0)]
```
Furthermore, there is 
```
u.vector *′ v.vector: 4x4-matrix:
⎛-1.0i       0.0  0.0  1.0 - 1.0i⎞
⎜-3.0i       0.0  0.0  3.0 - 3.0i⎟
⎜2.0         0.0  0.0  2.0 + 2.0i⎟
⎝4.0 - 1.0i  0.0  0.0  5.0 + 3.0i⎠
```
which is equivalent to 
```
Matrix(u.vector) * v.vector′: 4x4-matrix:
⎛-1.0i       0.0  0.0  1.0 - 1.0i⎞
⎜-3.0i       0.0  0.0  3.0 - 3.0i⎟
⎜2.0         0.0  0.0  2.0 + 2.0i⎟
⎝4.0 - 1.0i  0.0  0.0  5.0 + 3.0i⎠
```

## Solving Linear Equations

## Matrix Decompositions

## More

Complete documentation of the *LANumerics* API will eventually become available, but this is, just like the package itself, still work in progress. 
If you feel like experimenting and exploring more of the currently available functionality, examining the [current tests](https://github.com/phlegmaticprogrammer/LANumerics/tree/master/Tests/LANumericsTests) should provide a good starting point.

