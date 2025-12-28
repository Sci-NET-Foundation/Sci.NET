// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// Compute device representing a CPU with AVX and FMA support.
/// </summary>
public class AvxFmaCpuComputeDevice : CpuComputeDevice
{
    /// <inheritdoc />
    public override bool IsAvxFmaSupported()
    {
        if (!IntrinsicsHelper.IsAvxFmaSupported())
        {
            throw new NotSupportedException("The current CPU does not support AVX and FMA instructions.");
        }

        return true;
    }

    /// <inheritdoc />
    public override bool IsAvxSupported()
    {
        if (!IntrinsicsHelper.IsAvxSupported())
        {
            throw new NotSupportedException("The current CPU does not support AVX instructions.");
        }

        return true;
    }
}