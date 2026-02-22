// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.Intrinsics;
using Sci.NET.Mathematics.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.MicroKernels;

/// <summary>
/// A parameter container for micro-kernel operations.
/// </summary>
/// <typeparam name="TNumber">The numeric type.</typeparam>
public class MicroKernelParameter<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MicroKernelParameter{TNumber}"/> class with the specified scalar value.
    /// </summary>
    /// <param name="value">The scalar value.</param>
    public MicroKernelParameter(TNumber value)
    {
        ScalarValue = value;

        if (GenericMath.IsFloatingPoint<TNumber>())
        {
            ScalarFp32Value = float.CreateChecked(ScalarValue);
            ScalarFp64Value = double.CreateChecked(ScalarValue);
            Vector256ValueFp32 = Vector256.Create(ScalarFp32Value);
            Vector256ValueFp64 = Vector256.Create(ScalarFp64Value);
        }
        else
        {
            ScalarFp32Value = 0;
            ScalarFp64Value = 0;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MicroKernelParameter{TNumber}"/> class with the specified values.
    /// </summary>
    /// <param name="scalarValue">The scalar value.</param>
    /// <param name="scalarFp32Value">The scalar float (FP32) value.</param>
    /// <param name="scalarFp64Value">The scalar double (FP64) value.</param>
    /// <param name="vector256ValueFp32">The <see cref="Vector256{T}"/> float (FP32) value.</param>
    /// <param name="vector256ValueFp64">The <see cref="Vector256{T}"/> double (FP64) value.</param>
    public MicroKernelParameter(
        TNumber scalarValue,
        float scalarFp32Value,
        double scalarFp64Value,
        Vector256<float> vector256ValueFp32,
        Vector256<double> vector256ValueFp64)
    {
        ScalarValue = scalarValue;
        ScalarFp32Value = scalarFp32Value;
        ScalarFp64Value = scalarFp64Value;
        Vector256ValueFp32 = vector256ValueFp32;
        Vector256ValueFp64 = vector256ValueFp64;
    }

    /// <summary>
    /// Gets the scalar value.
    /// </summary>
    public TNumber ScalarValue { get; }

    /// <summary>
    /// Gets the scalar float (FP32) value.
    /// </summary>
    public float ScalarFp32Value { get; }

    /// <summary>
    /// Gets the scalar double (FP64) value.
    /// </summary>
    public double ScalarFp64Value { get; }

    /// <summary>
    /// Gets <see cref="Vector256{T}"/> float (FP32) value.
    /// </summary>
    public Vector256<float> Vector256ValueFp32 { get; }

    /// <summary>
    /// Gets the <see cref="Vector256{T}"/> float (FP64) value.
    /// </summary>
    public Vector256<double> Vector256ValueFp64 { get; }

    /// <summary>
    /// Implicitly converts a TNumber value to a <see cref="MicroKernelParameter{TNumber}"/>.
    /// </summary>
    /// <param name="value">The TNumber value.</param>
    /// <returns>The corresponding <see cref="MicroKernelParameter{TNumber}"/> instance.</returns>
    public static implicit operator MicroKernelParameter<TNumber>(TNumber value)
    {
        return new(value);
    }
}