// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedHypotBenchmarks<TNumber>
    where TNumber : unmanaged, IFloatingPointIeee754<TNumber>, IRootFunctions<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public Shape Shape { get; set; } = default!;

    public ICollection<Shape> ShapeOptions =>
    [
        new Shape(400, 200),
        new Shape(400, 200, 100),
        new Shape(400, 200, 100, 50)
    ];

    private ILinearAlgebraKernels _linearAlgebraKernels = default!;
    private Tensor<TNumber> _leftTensor = default!;
    private Tensor<TNumber> _rightTensor = default!;
    private Tensor<TNumber> _result = default!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        TNumber min;
        TNumber max;

        Tensor.SetDefaultBackend<ManagedTensorBackend>();

        if (GenericMath.IsFloatingPoint<TNumber>())
        {
            min = TNumber.CreateChecked(-1f);
            max = TNumber.CreateChecked(1f);
        }
        else if (GenericMath.IsSigned<TNumber>())
        {
            min = TNumber.CreateChecked(-10);
            max = TNumber.CreateChecked(10);
        }
        else
        {
            min = TNumber.CreateChecked(1);
            max = TNumber.CreateChecked(10);
        }

        _linearAlgebraKernels = ManagedTensorBackend.Instance.LinearAlgebra;
        _leftTensor = Tensor.Random.Uniform(Shape, min, max, seed: 123456).ToTensor();
        _rightTensor = Tensor.Random.Uniform(Shape, min, max, seed: 654321).ToTensor();
        _result = Tensor.Zeros<TNumber>(_leftTensor.Shape).ToTensor();
    }

    [Benchmark]
    public void Hypot()
    {
        _linearAlgebraKernels.Hypot(_leftTensor, _rightTensor, _result);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftTensor.Dispose();
        _rightTensor.Dispose();
        _result.Dispose();
    }
}