// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// Defines a unary parameterized operation supporting AVX2 with FMA instructions.
/// </summary>
/// <typeparam name="TSelf">The self type.</typeparam>
public interface IUnaryParameterizedOperationAvxFma<in TSelf> : IUnaryParameterizedOperationAvxTail<TSelf>
    where TSelf : IUnaryParameterizedOperationAvxFma<TSelf>
{
    /// <summary>
    /// Determines whether AVX2 with FMA is supported on the current hardware.
    /// </summary>
    /// <returns>A value indicating whether AVX2 with FMA is supported.</returns>
    public static abstract bool IsAvxFmaSupported();

    /// <summary>
    /// Invokes the vectorized operation on the input Vector256.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <param name="instance">The operation instance.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<float> ApplyAvxFmaFp32(Vector256<float> input, TSelf instance);

    /// <summary>
    /// Invokes the vectorized operation on the input Vector256.
    /// </summary>
    /// <param name="input">The left input.</param>
    /// <param name="instance">The operation instance.</param>
    /// <returns>The result of the operation.</returns>
    public static abstract Vector256<double> ApplyAvxFmaFp64(Vector256<double> input, TSelf instance);
}