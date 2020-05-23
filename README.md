# LANumerics

Copyright (c) 2020 Steven Obua

License: MIT License

*LANumerics* is a Swift package for doing *numerical linear algebra* in Swift. 

The package depends on [Swift Numerics](https://github.com/apple/swift-numerics), as it supports both **real** and **complex** numerics for both `Float` and `Double` precision in a uniform way. 
Under the hood it relies on the `Accelerate` framework for most of its functionality, in particular `BLAS` and `LAPACK`, and also `vDSP`.

## Usage
*LANumerics* is a normal Swift package and can be added to your App [in the usual way](https://developer.apple.com/documentation/xcode/adding_package_dependencies_to_your_app).
After adding it to your app, import `LANumerics` (and also `Numerics` if you use complex numbers). 

## LANumeric

The `LANumeric` protocol denotes the type of numbers on which *LANumerics* operates. It is implemented by the following types:

*  `Float`
*  `Double`
*  `Complex<Float>`
*  `Complex<Double>`

Most functionality of *LANumerics* is generic in `LANumeric`. Examples are solving a system of linear equations or computing the singular value decomposition of a matrix.

## Matrices

The main work horse data type of *LANumerics* is the `Matrix`. For convenience there is also the `Vector`, but this is just a typealias for normal Swift arrays.

## Solving Linear Equations

## Matrix Decompositions

## SIMD support
