// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedUnaryArithmeticBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public Shape Shape { get; set; } = default!;

    public ICollection<Shape> ShapeOptions =>
    [
        new Shape(400, 200),
        new Shape(400, 200, 100),
        new Shape(400, 200, 100, 50),
    ];

    private IArithmeticKernels _arithmeticKernels = default!;
    private Tensor<TNumber> _tensor = default!;
    private Tensor<TNumber> _result = default!;
    private Tensor<TNumber> _gradient = default!;

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

        _arithmeticKernels = ManagedTensorBackend.Instance.Arithmetic;
        _tensor = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 123456).ToTensor();
        _result = Tensor.Zeros<TNumber>(_tensor.Shape).ToTensor();
        _gradient = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 654321).ToTensor();
    }

    [Benchmark]
    public void Abs()
    {
        _arithmeticKernels.Abs(_tensor, _result);
    }

    [Benchmark]
    public void AbsBackwards()
    {
        _arithmeticKernels.AbsGradient(_tensor, _gradient, _result);
    }

    [Benchmark]
    public void Sqrt()
    {
        _arithmeticKernels.Sqrt(_tensor, _result);
    }

    [Benchmark]
    public void Negate()
    {
        _arithmeticKernels.Negate(_tensor, _result);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
        _gradient.Dispose();
    }
}