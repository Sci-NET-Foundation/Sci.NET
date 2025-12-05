// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Backends.Managed;

/// <summary>
/// An implementation of <see cref="ITensorBackend"/> for the managed backend.
/// </summary>
[PublicAPI]
public class ManagedTensorBackend : ITensorBackend
{
    internal const int ParallelizationThreshold = 10_000;

    /// <summary>
    /// Initializes a new instance of the <see cref="ManagedTensorBackend"/> class.
    /// </summary>
    public ManagedTensorBackend()
    {
        Storage = new ManagedStorageKernels();
        LinearAlgebra = new ManagedLinearAlgebraKernels();
        Arithmetic = new ManagedArithmeticKernels();
        Exponential = new ManagedExponentialKernels();
        Device = new CpuComputeDevice();
        Reduction = new ManagedReductionKernels();
        Trigonometry = new ManagedTrigonometryKernels();
        Random = new ManagedRandomKernels();
        Casting = new ManagedCastingKernels();
        ActivationFunctions = new ManagedActivationFunctionKernels();
        Broadcasting = new ManagedBroadcastingKernels();
        Permutation = new ManagedPermutationKernels();
        Normalisation = new ManagedNormalisationKernels();
        EqualityOperations = new ManagedEqualityOperationKernels();
    }

    /// <summary>
    /// Gets the maximum degree of parallelism for operations in the managed backend.
    /// </summary>
    public static int MaxDegreeOfParallelism { get; private set; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets the singleton instance of the <see cref="ManagedTensorBackend"/>.
    /// </summary>
    public static ITensorBackend Instance { get; } = new ManagedTensorBackend();

    /// <inheritdoc />
    public ITensorStorageKernels Storage { get; }

    /// <inheritdoc />
    public ILinearAlgebraKernels LinearAlgebra { get; }

    /// <inheritdoc />
    public IArithmeticKernels Arithmetic { get; }

    /// <inheritdoc />
    public IExponentialKernels Exponential { get; }

    /// <inheritdoc />
    public IDevice Device { get; }

    /// <inheritdoc />
    public IReductionKernels Reduction { get; }

    /// <inheritdoc />
    public ITrigonometryKernels Trigonometry { get; }

    /// <inheritdoc />
    public IRandomKernels Random { get; }

    /// <inheritdoc />
    public ICastingKernels Casting { get; }

    /// <inheritdoc />
    public IActivationFunctionKernels ActivationFunctions { get; }

    /// <inheritdoc />
    public IBroadcastingKernels Broadcasting { get; }

    /// <inheritdoc />
    public IPermutationKernels Permutation { get; }

    /// <inheritdoc />
    public INormalisationKernels Normalisation { get; }

    /// <inheritdoc />
    public IEqualityOperationKernels EqualityOperations { get; }

    /// <summary>
    /// Sets the maximum degree of parallelism for operations in the managed backend.
    /// </summary>
    /// <param name="degreeOfParallelism">The maximum degree of parallelism to set.</param>
    public static void SetMaxDegreeOfParallelism(int degreeOfParallelism)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(degreeOfParallelism, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(degreeOfParallelism, Environment.ProcessorCount);

        MaxDegreeOfParallelism = degreeOfParallelism;
    }

    internal static int GetMaxDegreeOfParallelism(long tileCount)
    {
        if (tileCount > int.MaxValue)
        {
            return MaxDegreeOfParallelism;
        }

        return Math.Min(MaxDegreeOfParallelism, (int)tileCount);
    }

    internal static bool ShouldParallelizeForTiles(long tileCount)
    {
        // Arbitrary threshold of 2 tiles to decide whether to parallelize
        return tileCount > 2;
    }

    internal static int GetNumThreadsByElementCount<TNumber>(long elementCount)
        where TNumber : unmanaged, INumber<TNumber>
    {
        const long minBytesPerThread = 256 * 1024; // 256 KB per thread
        var maxUsefulThreads = Math.Max(1, elementCount * Unsafe.SizeOf<TNumber>() / minBytesPerThread);

        return (int)Math.Min(maxUsefulThreads, MaxDegreeOfParallelism);
    }

    internal static bool ShouldStream(long elementCount)
    {
        const long streamThreshold = 10_000; // Arbitrary threshold for streaming

        return elementCount >= streamThreshold;
    }
}