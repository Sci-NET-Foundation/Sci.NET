// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Mathematics.Memory;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Random;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedRandomKernels : IRandomKernels
{
    private readonly Prng _prng = Prng.Instance;

    public void Seed(ulong value)
    {
        _prng.SetSeed(value);
    }

    public ITensor<TNumber> Uniform<TNumber>(Shape shape, TNumber min, TNumber max, ulong? seed = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Tensor<TNumber>(shape, ManagedTensorBackend.Instance);
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                BFloat16.CreateChecked(min),
                BFloat16.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                Half.CreateChecked(min),
                Half.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                float.CreateChecked(min),
                float.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
                double.CreateChecked(min),
                double.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(sbyte))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<sbyte>>(memoryBlock),
                sbyte.CreateChecked(min),
                sbyte.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(byte))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<byte>>(memoryBlock),
                byte.CreateChecked(min),
                byte.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(ushort))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<ushort>>(memoryBlock),
                ushort.CreateChecked(min),
                ushort.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(short))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<short>>(memoryBlock),
                short.CreateChecked(min),
                short.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(uint))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<uint>>(memoryBlock),
                uint.CreateChecked(min),
                uint.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(int))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<int>>(memoryBlock),
                int.CreateChecked(min),
                int.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(ulong))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<ulong>>(memoryBlock),
                ulong.CreateChecked(min),
                ulong.CreateChecked(max));
        }
        else if (typeof(TNumber) == typeof(long))
        {
            prng.FillUniform(
                Unsafe.As<SystemMemoryBlock<long>>(memoryBlock),
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
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(Half))
        {
            prng.FillNormal(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                Half.CreateChecked(mean),
                Half.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillNormal(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                BFloat16.CreateChecked(mean),
                BFloat16.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillNormal(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                float.CreateChecked(mean),
                float.CreateChecked(stdDev));
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillNormal(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
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
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(Half))
        {
            prng.FillXavierUniform(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillXavierUniform(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillXavierUniform(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillXavierUniform(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
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
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillXavierNormal(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillXavierNormal(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillXavierNormal(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                inputUnits,
                outputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillXavierNormal(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
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
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillHeUniform(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillHeUniform(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillHeUniform(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillHeUniform(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
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
        var prng = seed is null ? _prng : new Prng(seed.Value);
        var memoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;

        if (typeof(TNumber) == typeof(BFloat16))
        {
            prng.FillHeNormal(
                Unsafe.As<SystemMemoryBlock<BFloat16>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(Half))
        {
            prng.FillHeNormal(
                Unsafe.As<SystemMemoryBlock<Half>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(float))
        {
            prng.FillHeNormal(
                Unsafe.As<SystemMemoryBlock<float>>(memoryBlock),
                inputUnits);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            prng.FillHeNormal(
                Unsafe.As<SystemMemoryBlock<double>>(memoryBlock),
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