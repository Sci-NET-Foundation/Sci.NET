// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Backends.Managed.Iterators;
using Sci.NET.Mathematics.Backends.Managed.MicroKernels.Arithmetic;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedNormalisationKernels : INormalisationKernels
{
    public unsafe void Clip<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new ClampMicroKernel<TNumber>(min, max),
            tensor.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }

    public unsafe void ClipBackward<TNumber>(ITensor<TNumber> tensor, Tensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        ManagedUnaryOperationIterator.Apply(
            tensor.Memory.ToPointer(),
            result.Memory.ToPointer(),
            new ClampBackwardMicroKernel<TNumber>(min, max),
            tensor.Memory.Length,
            (ICpuComputeDevice)tensor.Device);
    }
}