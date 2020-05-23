# LANumerics

Copyright (c) 2020 Steven Obua

License: MIT License

*LANumerics* is a Swift package for doing *numerical linear algebra* in Swift. 

The package depends on [Swift Numerics](https://github.com/apple/swift-numerics), as it supports both **real** and **complex** numerics for both `Float` and `Double` precision in a uniform way. 
Under the hood it relies on the `Accelerate` framework for most of its functionality, in particular `BLAS` and `LAPACK`, and also `vDSP`.

