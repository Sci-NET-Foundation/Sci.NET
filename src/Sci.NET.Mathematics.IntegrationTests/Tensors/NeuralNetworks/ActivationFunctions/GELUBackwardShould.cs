// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class GELUBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnCorrectValues_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -50, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 60 });

        value.To(device);

        // Act
        var result = value.GELUBackward();

        // Assert
        result
            .Should()
            .HaveShape(12)
            .And
            .HaveApproximatelyEquivalentElements(
                new float[] { 0F, -0.0003351313F, -0.011584155F, -0.086099304F, -0.08296409F, 0.5F, 1.0829642F, 1.0860993F, 1.0115842F, 1.0003352F, 1.0000015F, 1F },
                1e-6f);
    }
}