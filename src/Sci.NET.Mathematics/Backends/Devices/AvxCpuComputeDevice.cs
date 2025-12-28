// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// Represents a CPU compute device with AVX support.
/// </summary>
public class AvxCpuComputeDevice : CpuComputeDevice
{
    /// <inheritdoc />
    public override bool IsAvxFmaSupported()
    {
        return false;
    }

    /// <inheritdoc />
    public override bool IsAvxSupported()
    {
        if (!IntrinsicsHelper.IsAvxSupported())
        {
            throw new NotSupportedException("AVX is not supported on this CPU.");
        }

        return true;
    }
}