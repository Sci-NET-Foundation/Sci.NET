// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedBinaryArithmeticBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public (Shape LeftShape, Shape RightShape) Shapes { get; set; } = default!;

    public ICollection<(Shape LeftShape, Shape RightShape)> ShapeOptions =>
    [
        (new Shape(400, 200), new Shape(400, 200)),
        (new Shape(400, 200, 100), new Shape(200, 100)),
        (new Shape(400, 200, 100, 50), new Shape(200, 100, 50)),
    ];

    private IArithmeticKernels _arithmeticKernels = default!;
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

        _arithmeticKernels = ManagedTensorBackend.Instance.Arithmetic;
        _leftTensor = Tensor.Random.Uniform<TNumber>(Shapes.LeftShape, min, max, seed: 123456).ToTensor();
        _rightTensor = Tensor.Random.Uniform<TNumber>(Shapes.RightShape, min, max, seed: 654321).ToTensor();
        _result = Tensor.Zeros<TNumber>(_leftTensor.Shape).ToTensor();
    }

    [Benchmark]
    public void Add()
    {
        _arithmeticKernels.Add(_leftTensor, _rightTensor, _result);
    }

    [Benchmark]
    public void Subtract()
    {
        _arithmeticKernels.Subtract(_leftTensor, _rightTensor, _result);
    }

    [Benchmark]
    public void Multiply()
    {
        _arithmeticKernels.Multiply(_leftTensor, _rightTensor, _result);
    }

    [Benchmark]
    public void Divide()
    {
        _arithmeticKernels.Divide(_leftTensor, _rightTensor, _result);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftTensor.Dispose();
        _rightTensor.Dispose();
        _result.Dispose();
    }
}