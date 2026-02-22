// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedReductionBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public Shape Shape { get; set; } = default!;

    public ICollection<Shape> ShapeOptions =>
    [
        new Shape(32, 32, 32),
        new Shape(64, 64, 64),
        new Shape(32, 32, 32, 32),
        new Shape(64, 64, 64, 64)
    ];

    private Tensor<TNumber> _tensor = default!;
    private ITensor<TNumber> _result = default!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        TNumber min;
        TNumber max;

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

        _tensor = Tensor.Random.Uniform(Shape, min, max, seed: 123456).ToTensor();
    }

    [Benchmark]
    public void SumAll()
    {
        _result = _tensor.Sum();
    }

    [Benchmark]
    public void SumOutermost()
    {
        _result = _tensor.Sum(axes: [0]);
    }

    [Benchmark]
    public void SumInnermost()
    {
        _result = _tensor.Sum(axes: [Shape.Rank - 1]);
    }

    [Benchmark]
    public void SumMiddle()
    {
        var middleAxis = Shape.Rank / 2;
        _result = _tensor.Sum(axes: [middleAxis]);
    }

    [Benchmark]
    public void MeanAll()
    {
        _result = _tensor.Mean();
    }

    [Benchmark]
    public void MeanOutermost()
    {
        _result = _tensor.Mean(axes: [0]);
    }

    [Benchmark]
    public void MeanInnermost()
    {
        _result = _tensor.Mean(axes: [Shape.Rank - 1]);
    }

    [Benchmark]
    public void MeanMiddle()
    {
        var middleAxis = Shape.Rank / 2;
        _result = _tensor.Mean(axes: [middleAxis]);
    }

    [Benchmark]
    public void MaxAll()
    {
        _result = _tensor.Max();
    }

    [Benchmark]
    public void MinOutermost()
    {
        _result = _tensor.Min(axes: [0]);
    }

    [Benchmark]
    public void MinInnermost()
    {
        _result = _tensor.Min(axes: [Shape.Rank - 1]);
    }

    [Benchmark]
    public void MinMiddle()
    {
        var middleAxis = Shape.Rank / 2;
        _result = _tensor.Min(axes: [middleAxis]);
    }

    [Benchmark]
    public void MinAll()
    {
        _result = _tensor.Min();
    }

    [Benchmark]
    public void MaxOutermost()
    {
        _result = _tensor.Max(axes: [0]);
    }

    [Benchmark]
    public void MaxInnermost()
    {
        _result = _tensor.Max(axes: [Shape.Rank - 1]);
    }

    [Benchmark]
    public void MaxMiddle()
    {
        _result = _tensor.Max(axes: [Shape.Rank / 2]);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
    }
}