// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Backends.Devices;

/// <summary>
/// A CPU compute device interface.
/// </summary>
public interface ICpuComputeDevice : IDevice
{
    /// <summary>
    /// Determines whether the CPU supports AVX2 instructions.
    /// </summary>
    /// <returns>True if AVX2 is supported; otherwise, false.</returns>
    public bool IsAvx2Supported();
}