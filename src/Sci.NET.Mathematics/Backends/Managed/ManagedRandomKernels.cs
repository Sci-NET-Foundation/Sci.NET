// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Random;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedRandomKernels : IRandomKernels
{
    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, ulong? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (seed is not null)
        {
            prng.SetSeed(seed.Value);
        }

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                BFloat16.CreateChecked(min),
                BFloat16.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<Half>(),
                Half.CreateChecked(min),
                Half.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<float>(),
                float.CreateChecked(min),
                float.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<double>(),
                double.CreateChecked(min),
                double.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(sbyte))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<sbyte>(),
                sbyte.CreateChecked(min),
                sbyte.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(byte))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<byte>(),
                byte.CreateChecked(min),
                byte.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(ushort))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<ushort>(),
                ushort.CreateChecked(min),
                ushort.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(short))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<short>(),
                short.CreateChecked(min),
                short.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(uint))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<uint>(),
                uint.CreateChecked(min),
                uint.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(int))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<int>(),
                int.CreateChecked(min),
                int.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(ulong))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<ulong>(),
                ulong.CreateChecked(min),
                ulong.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(long))
        {
            prng.FillUniform(
                memoryBlock.DangerousReinterpretCast<long>(),
                long.CreateChecked(min),
                long.CreateChecked(max));
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }

    public ITensor<TNumber> Normal<TNumber>(Shape shape, TNumber mean, TNumber stdDev, ulong? seed = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(Half))
        {
            prng.FillNormal(
                memoryBlock.DangerousReinterpretCast<Half>(),
                Half.CreateChecked(mean),
                Half.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillNormal(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                BFloat16.CreateChecked(mean),
                BFloat16.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillNormal(
                memoryBlock.DangerousReinterpretCast<float>(),
                float.CreateChecked(mean),
                float.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillNormal(
                memoryBlock.DangerousReinterpretCast<double>(),
                double.CreateChecked(mean),
                double.CreateChecked(stdDev));
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }

    public ITensor<TNumber> XavierUniform<TNumber>(Shape shape, int inputUnits, int outputUnits, ulong? seed = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(Half))
        {
            prng.FillXavierUniform(
                memoryBlock.DangerousReinterpretCast<Half>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillXavierUniform(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillXavierUniform(
                memoryBlock.DangerousReinterpretCast<float>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillXavierUniform(
                memoryBlock.DangerousReinterpretCast<double>(),
                inputUnits,
                outputUnits);
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }

    public ITensor<TNumber> XavierNormal<TNumber>(Shape shape, int inputUnits, int outputUnits, ulong? seed = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillXavierNormal(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillXavierNormal(
                memoryBlock.DangerousReinterpretCast<Half>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillXavierNormal(
                memoryBlock.DangerousReinterpretCast<float>(),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillXavierNormal(
                memoryBlock.DangerousReinterpretCast<double>(),
                inputUnits,
                outputUnits);
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }

    public ITensor<TNumber> HeUniform<TNumber>(Shape shape, int inputUnits, ulong? seed = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillHeUniform(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillHeUniform(
                memoryBlock.DangerousReinterpretCast<Half>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillHeUniform(
                memoryBlock.DangerousReinterpretCast<float>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillHeUniform(
                memoryBlock.DangerousReinterpretCast<double>(),
                inputUnits);
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }

    public ITensor<TNumber> HeNormal<TNumber>(Shape shape, int inputUnits, ulong? seed = null)
        where TNumber : unmanaged, IFloatingPoint<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? Prng.Instance : new Prng();
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillHeNormal(
                memoryBlock.DangerousReinterpretCast<BFloat16>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillHeNormal(
                memoryBlock.DangerousReinterpretCast<Half>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillHeNormal(
                memoryBlock.DangerousReinterpretCast<float>(),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillHeNormal(
                memoryBlock.DangerousReinterpretCast<double>(),
                inputUnits);
        }
        else
        {
            tensor.Dispose();
            throw new NotSupportedException($"Type {typeof(TNumber)} is not supported.");
        }

        return tensor;
    }
}