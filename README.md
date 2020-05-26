# LANumerics ![](https://github.com/phlegmaticprogrammer/LANumerics/workflows/Build%20%26%20Test/badge.svg)

Copyright (c) 2020 Steven Obua

License: MIT License

*LANumerics* is a Swift package for doing *numerical linear algebra*. 

The package depends on [Swift Numerics](https://github.com/apple/swift-numerics), as it supports both **real** and **complex** numerics for both `Float` and `Double` precision in a uniform way. 
Under the hood it relies on the [`Accelerate`](https://developer.apple.com/documentation/accelerate) framework for most of its functionality, in particular `BLAS` and `LAPACK`, and also `vDSP`.

## Table of Contents

* [Usage](#usage)
* [LANumeric](#lanumeric)
* [Constructing Matrices](#constructing-matrices)
* [SIMD Support](#simd-support)
* [Matrix Arithmetic](#matrix-arithmetic)
* [Solving Linear Equations](#solving-linear-equations)
* [Linear Least Squares](#linear-least-squares)
* [Matrix Decompositions](#matrix-decompositions)

Complete documentation of the *LANumerics* API will eventually become available, but this is, just like the package itself, still work in progress. 
If you feel like experimenting and exploring more of the currently available functionality, examining the [current tests](https://github.com/phlegmaticprogrammer/LANumerics/tree/master/Tests/LANumericsTests) could provide a good starting point beyond this README.

---

## Usage
*LANumerics* is a normal Swift package and can be added to your app [in the usual way](https://developer.apple.com/documentation/xcode/adding_package_dependencies_to_your_app).
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

Most functionality of *LANumerics* is generic in `LANumeric`, e.g. constructing matrices and computing with them, solving a system of linear equations, or computing the singular value decomposition of a matrix.

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

Swift supports `simd` vector and matrix operations. *LANumerics* plays nice with `simd` by providing conversion functions to and from `simd` vectors and matrices. For example, starting from
```swift
import simd
import LANumerics

let m = Matrix(rows: [[1, 2, 3], [4, 5, 6]])
print("m: \(m)")
```
with output 
```
m: 2x3-matrix:
⎛1.0  2.0  3.0⎞
⎝4.0  5.0  6.0⎠
```
we can convert `m` into a `simd` matrix `s` via
```swift
let s = m.simd3x2
print(s)
```
resulting in the output 
```
simd_double3x2(columns: (SIMD2<Double>(1.0, 4.0), SIMD2<Double>(2.0, 5.0), SIMD2<Double>(3.0, 6.0)))
```
Note that `simd` reverses the role of row and column indices compared to `LANumerics` (and usual mathematical convention).

We can also convert `s` back:
```swift
print(Matrix(s) == m)
```
will yield the output `true`.

## Accessing Matrix Elements and Submatrices

Matrix elements and submatrices can be accessed using familiar notation. Given
```swift
import simd
import LANumerics

var m = Matrix(rows: [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m)
```
with output
```
3x3-matrix:
⎛1.0  2.0  3.0⎞
⎜4.0  5.0  6.0⎟
⎝7.0  8.0  9.0⎠
```
we can access the element at row 2 and column 1 via
```
m[2, 1]
```
which yields `8.0`. We can also set the element at row 2 and column 1 to some value:
```
m[2, 1] = 0
print(m)
```
The output of running this is
```
3x3-matrix:
⎛1.0  2.0  3.0⎞
⎜4.0  5.0  6.0⎟
⎝7.0  0.0  9.0⎠
```
We can also access submatrices of `m`, for example its top-left and bottom-right 2x2 submatrices:
```
print(m[0 ... 1, 0 ... 1])
print(m[1 ... 2, 1 ... 2])

```
This will print
```
2x2-matrix:
⎛1.0  2.0⎞
⎝4.0  5.0⎠
2x2-matrix:
⎛5.0  6.0⎞
⎝0.0  9.0⎠
```
Finally, using the same notation, we can overwrite submatrices of `m`:
```
m[0 ... 1, 0 ... 1] = m[1 ... 2, 1 ... 2]
print(m)
```
This overwrites the top-left 2x2 submatrix of `m` with its bottom-right 2x2 submatrix, yielding:
```
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
* element-wise operations
* functional operations

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

Note that `′` is the unicode character "Prime" `U+2032`. You can use for example [Ukelele](https://software.sil.org/ukelele/) to make the input of that character smooth. Other alternatives are configuring the touchbar of your macbook, or using a configurable keyboard like [Stream Deck](https://www.elgato.com/en/gaming/stream-deck).


### Matrix Multiplication

Multiplying `u` and `v` is done via the expression `u * v`. Running `print("u * v: \(u * v)")` results in
```
u * v: 2x2-matrix:
⎛1.0i  -2.0 + 2.0i⎞
⎝3.0i  -3.0 + 5.0i⎠
```

Instead of `u′ * v` one can also use the equivalent, but faster expression `u ′* v`:
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
u.vector: [1.0, 3.0, 2.0i, 1.0 + 4.0i]
v.vector: [1.0i, 0.0, 0.0, 1.0 + 1.0i]
```
(Actually, in the above, we used `u.vector.toString()` and `v.vector.toString()` for better formatting of complex numbers. We will also do so below where appropriate without further mentioning it.)

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
u.vector′ * v.vector: [5.0 - 2.0i]
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

### Element-wise Operations

Element-wise operations like `.+`, `.-`, `.*` and `./` are supported on both vectors and matrices, for example:

```
u .* v : 2x2-matrix:
⎛1.0i  0.0        ⎞
⎝0.0   -3.0 + 5.0i⎠
```

### Functional Operations

The `Matrix` type supports functional operations like `map`, `reduce` and `combine`. These come in handy when performance is not that important, and there is no accelerated equivalent available (yet?).
For example, the expression
```swift
u.reduce(0) { x, y in max(x, y.magnitude) }
```
results in the value `4`. In this case it is better though to use the equivalent expression `u.infNorm` instead.

## Solving Linear Equations

You can solve a system of linear equations like 

1.  7x+5y-3z = 16
2.  3x-5y+2z = -8
3.  5x+3y-7z = 0

by converting it into matrix form `A * u = b` and solving it for `u`:
```swift
import LANumerics
let A = Matrix<Double>(rows: [[7, 5, -3], [3, -5, 2], [5, 3, -7]])
let b : Vector<Double> = [16, -8, 0]
let u = A.solve(b)!
print("A: \(A)\n")
print("b: \(b)\n")
print("u: \(u.toString(precision: 1))")
```
This results in the output:
```
A: 3x3-matrix:
⎛7.0  5.0   -3.0⎞
⎜3.0  -5.0  2.0 ⎟
⎝5.0  3.0   -7.0⎠

b: [16.0, -8.0, 0.0]

u: [1.0, 3.0, 2.0]
```
Therefore the solution is x=1, y=3, z=2.

This example is actually plugged from an [article](https://www.appcoda.com/accelerate-framework/) which describes how to use the `Accelerate` framework directly and without a nice library like `LANumerics` 😁.

You can solve for multiple right-hand sides simultaneously. For example, you can compute the inverse of `A` like so:
```swift
let Id : Matrix<Double> = .eye(3)
print("Id: \(Id)\n")
let U = A.solve(Id)!
print("U: \(U)\n")
print("A * U: \(A * U)")
```
This results in the output
```
Id: 3x3-matrix:
⎛1.0  0.0  0.0⎞
⎜0.0  1.0  0.0⎟
⎝0.0  0.0  1.0⎠

U: 3x3-matrix:
⎛0.11328125  0.1015625             -0.019531249999999997⎞
⎜0.12109375  -0.13281250000000003  -0.08984375          ⎟
⎝0.1328125   0.01562499999999999   -0.1953125           ⎠

A * U: 3x3-matrix:
⎛1.0  -1.0755285551056204e-16  0.0⎞
⎜0.0  1.0                      0.0⎟
⎝0.0  -4.163336342344337e-17   1.0⎠
```
## Linear Least Squares

Extending the above example, you can also solve it using *least squares approximation* instead:
```swift
print(A.solveLeastSquares(b)!.toString(precision: 1))
```
results in the same solution `[1.0, 3.0, 2.0]`. 

Least squares is more general than solving linear equations directly, as it can also deal with situations where you have more equations than variables, or less equations than variables. In other words, it can also handle:
* non-square matrices `A` 
* situations with large noise in the data 
* situations which are only approximately linear

There is a shorthand notation available for the expression `A.solveLeastSquares(b)!`:
```swift
A ∖ b
```
This also works for simultaneously solving for multiple right-hand sides as before, the inverse of `A` can therefore also be computed using the expression `A ∖ .eye(3)`:
```
A ∖ .eye(3): 3x3-matrix:
⎛0.11328124999999997  0.10156250000000001   -0.01953124999999994⎞
⎜0.12109375           -0.13281250000000006  -0.08984374999999999⎟
⎝0.13281249999999997  0.015624999999999993  -0.19531249999999994⎠
```
Note that `∖` is the unicode character "Set Minus"  `U+2216`. The same advice for smooth input of this character applies as for the input of `′` earlier.

In addition, there is the operator `′∖` which combines taking the adjoint and solving via least squares. Therefore, to compute the inverse of `A′`, you could better write `A ′∖ .eye(3)` instead of `A′ ∖ .eye(3)`:
```
A ′∖ .eye(3): 3x3-matrix:
⎛0.11328124999999996   0.12109375            0.13281249999999994 ⎞
⎜0.10156250000000001   -0.13281250000000003  0.01562499999999999 ⎟
⎝-0.01953124999999994  -0.08984374999999999  -0.19531249999999994⎠
```

## Matrix Decompositions

The following matrix decompositions are currently supported:

* Singular value decomposition of a real or complex matrix `A`: 
  ```swift
  A.svd()
  ```
* Eigen decomposition for self-adjoint matrices `A` (that is for real or complex matrices for which `A == A′`):
  ```swift
  A.eigen()
  ```
* Schur decomposition of a real or complex square-matrix `A`: 
  ```swift
  A.schur()
  ```


