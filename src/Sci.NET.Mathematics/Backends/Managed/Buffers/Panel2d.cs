// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed.Buffers;

internal readonly struct Panel2d<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    public readonly unsafe TNumber* A;
    public readonly unsafe TNumber* B;

    public unsafe Panel2d(TNumber* a, TNumber* b)
    {
        A = a;
        B = b;
    }
}