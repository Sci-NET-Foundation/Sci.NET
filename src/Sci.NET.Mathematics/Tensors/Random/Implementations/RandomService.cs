// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors.Random.Implementations;

internal class RandomService : IRandomService
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.Uniform(
                shape,
                min,
                max,
                seed,
                device);
    }

    public ITensor<TNumber> Normal<TNumber>(Shape shape, TNumber mean, TNumber stdDev, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.Normal(
                shape,
                mean,
                stdDev,
                seed,
                device);
    }

    public ITensor<TNumber> XavierUniform<TNumber>(Shape shape, int inputUnits, int outputUnits, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.XavierUniform<TNumber>(
                shape,
                inputUnits,
                outputUnits,
                seed,
                device);
    }

    public ITensor<TNumber> XavierNormal<TNumber>(Shape shape, int inputUnits, int outputUnits, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.XavierNormal<TNumber>(
                shape,
                inputUnits,
                outputUnits,
                seed,
                device);
    }

    public ITensor<TNumber> HeUniform<TNumber>(Shape shape, int inputUnits, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.HeUniform<TNumber>(
                shape,
                inputUnits,
                seed,
                device);
    }

    public ITensor<TNumber> HeNormal<TNumber>(Shape shape, int inputUnits, ulong? seed = null, IDevice? device = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        device ??= Tensor.DefaultBackend.Device;

        return device
            .GetTensorBackend()
            .Random.HeNormal<TNumber>(
                shape,
                inputUnits,
                seed,
                device);
    }

    public void Seed(ulong seed, IDevice? device = null)
    {
        if (device is not null)
        {
            device
                .GetTensorBackend()
                .Random.Seed(seed);
        }
        else
        {
            Tensor.DefaultBackend
                .Random.Seed(seed);
        }
    }
}