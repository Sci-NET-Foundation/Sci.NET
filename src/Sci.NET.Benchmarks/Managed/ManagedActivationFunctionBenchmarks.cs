// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

[MaxIterationCount(4096)]
public class ManagedActivationFunctionBenchmarks<TNumber>
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

    private IActivationFunctionKernels _activationFunctionKernels = default!;
    private Tensor<TNumber> _tensor = default!;
    private ITensor<TNumber> _result = default!;
    private TNumber _alpha;
    private TNumber _min;
    private TNumber _max;

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
            _alpha = TNumber.CreateChecked(0.01f); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(-1f); // Hard Tanh min
            _max = TNumber.CreateChecked(1f); // Hard Tanh max
        }
        else if (GenericMath.IsSigned<TNumber>())
        {
            min = TNumber.CreateChecked(-10);
            max = TNumber.CreateChecked(10);
            _alpha = TNumber.CreateChecked(1); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(-10); // Hard Tanh min
            _max = TNumber.CreateChecked(10); // Hard Tanh max
        }
        else
        {
            min = TNumber.CreateChecked(1);
            max = TNumber.CreateChecked(10);
            _alpha = TNumber.CreateChecked(0.01f); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(1); // Hard Tanh min
            _max = TNumber.CreateChecked(10); // Hard Tanh max
        }

        _tensor = Tensor.Random.Uniform(Shape, min, max, seed: 123456).ToTensor();
        _result = Tensor.Zeros<TNumber>(Shape);
        _activationFunctionKernels = ManagedTensorBackend.Instance.ActivationFunctions;
    }

    [Benchmark]
    public void ReLU()
    {
        _activationFunctionKernels.ReLU(_tensor, _result);
    }

    [Benchmark]
    public void ReLUBackward()
    {
        _activationFunctionKernels.ReLUBackward(_tensor, _result);
    }

    [Benchmark]
    public void LeakyReLU()
    {
        _activationFunctionKernels.LeakyReLU(_tensor, _result, _alpha);
    }

    [Benchmark]
    public void LeakyReLUBackward()
    {
        _activationFunctionKernels.LeakyReLUBackward(_tensor, _result, _alpha);
    }

    [Benchmark]
    public void SoftSign()
    {
        _activationFunctionKernels.SoftSign(_tensor, _result);
    }

    [Benchmark]
    public void SoftSignBackward()
    {
        _activationFunctionKernels.SoftSignBackward(_tensor, _result);
    }

    [Benchmark]
    public void HardSigmoid()
    {
        _activationFunctionKernels.HardSigmoid(_tensor, _result);
    }

    [Benchmark]
    public void HardSigmoidBackward()
    {
        _activationFunctionKernels.HardSigmoidBackward(_tensor, _result);
    }

    [Benchmark]
    public void HardTanh()
    {
        _activationFunctionKernels.HardTanh(_tensor, _result, _min, _max);
    }

    [Benchmark]
    public void HardTanhBackward()
    {
        _activationFunctionKernels.HardTanhBackward(_tensor, _result, _min, _max);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
    }
}