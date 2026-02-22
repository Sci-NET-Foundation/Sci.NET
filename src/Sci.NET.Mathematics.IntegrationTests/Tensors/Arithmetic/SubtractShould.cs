// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Numerics;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class SubtractShould : IntegrationTestBase, IArithmeticTests
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device)
    {
        ScalarScalarTest<BFloat16>(2, 1, device).Should().Be(1);
        ScalarScalarTest<float>(2, 1, device).Should().Be(1);
        ScalarScalarTest<double>(2, 1, device).Should().Be(1);
        ScalarScalarTest<byte>(2, 1, device).Should().Be(1);
        ScalarScalarTest<sbyte>(2, 1, device).Should().Be(1);
        ScalarScalarTest<ushort>(2, 1, device).Should().Be(1);
        ScalarScalarTest<short>(2, 1, device).Should().Be(1);
        ScalarScalarTest<uint>(2, 1, device).Should().Be(1);
        ScalarScalarTest(2, 1, device).Should().Be(1);
        ScalarScalarTest<long>(2, 1, device).Should().Be(1);
        ScalarScalarTest<ulong>(2, 1, device).Should().Be(1);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightScalar = new Scalar<TNumber>(right);
        leftScalar.To(device);
        rightScalar.To(device);

        var result = leftScalar.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device)
    {
        ScalarVectorTest(5, new BFloat16[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new BFloat16[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest(5, new float[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new float[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest(5, new double[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new double[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<byte>(5, new byte[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new byte[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<sbyte>(5, new sbyte[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new sbyte[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<ushort>(5, new ushort[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new ushort[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<short>(5, new short[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new short[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<uint>(5, new uint[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new uint[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest(5, new int[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new int[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<ulong>(5, new ulong[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new ulong[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest(5, new long[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new long[] { 4, 3, 2, 1, 0 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device)
    {
        ScalarMatrixTest(5, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest(5, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest(5, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<byte>(5, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<sbyte>(5, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<ushort>(5, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<short>(5, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<uint>(5, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest(5, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<ulong>(5, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest(5, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 4, 3 }, { 2, 1 } });
    }

    private static Array ScalarMatrixTest<TNumber>(TNumber left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To(device);
        rightMatrix.To(device);

        var result = leftScalar.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        ScalarTensorTest(5, new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest(5, new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest(5, new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<byte>(5, new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<sbyte>(5, new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<ushort>(5, new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<short>(5, new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<uint>(5, new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest(5, new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<ulong>(5, new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest(5, new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
    }

    private static Array ScalarTensorTest<TNumber>(TNumber left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftScalar.To(device);
        rightTensor.To(device);

        var result = leftScalar.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device)
    {
        VectorScalarTest(new BFloat16[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new BFloat16[] { 5, 4, 3, 2, 1 });
        VectorScalarTest(new float[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new float[] { 5, 4, 3, 2, 1 });
        VectorScalarTest(new double[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new double[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<byte>(new byte[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new byte[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<sbyte>(new sbyte[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new sbyte[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<ushort>(new ushort[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new ushort[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<short>(new short[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new short[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<uint>(new uint[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new uint[] { 5, 4, 3, 2, 1 });
        VectorScalarTest(new int[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new int[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<ulong>(new ulong[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new ulong[] { 5, 4, 3, 2, 1 });
        VectorScalarTest(new long[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new long[] { 5, 4, 3, 2, 1 });
    }

    private static Array VectorScalarTest<TNumber>(TNumber[] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightScalar = new Scalar<TNumber>(right);
        leftVector.To(device);
        rightScalar.To(device);

        var result = leftVector.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device)
    {
        VectorVectorTest(new BFloat16[] { 7, 6, 5, 4, 3 }, new BFloat16[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new BFloat16[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new float[] { 7, 6, 5, 4, 3 }, new float[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new float[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new double[] { 7, 6, 5, 4, 3 }, new double[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new double[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new byte[] { 7, 6, 5, 4, 3 }, new byte[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new byte[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new sbyte[] { 7, 6, 5, 4, 3 }, new sbyte[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new sbyte[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new ushort[] { 7, 6, 5, 4, 3 }, new ushort[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new ushort[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new short[] { 7, 6, 5, 4, 3 }, new short[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new short[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new uint[] { 7, 6, 5, 4, 3 }, new uint[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new uint[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new int[] { 7, 6, 5, 4, 3 }, new int[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new int[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new ulong[] { 7, 6, 5, 4, 3 }, new ulong[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new ulong[] { 5, 4, 3, 2, 1 });
        VectorVectorTest(new long[] { 7, 6, 5, 4, 3 }, new long[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new long[] { 5, 4, 3, 2, 1 });
    }

    private static Array VectorVectorTest<TNumber>(TNumber[] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftVector.To(device);
        rightVector.To(device);

        var result = leftVector.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device)
    {
        VectorMatrixTest(new BFloat16[] { 7, 6 }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new float[] { 7, 6 }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new double[] { 7, 6 }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new byte[] { 7, 6 }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new sbyte[] { 7, 6 }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new ushort[] { 7, 6 }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new short[] { 7, 6 }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new uint[] { 7, 6 }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new int[] { 7, 6 }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new ulong[] { 7, 6 }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest(new long[] { 7, 6 }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 6, 4 }, { 4, 2 } });
    }

    private static Array VectorMatrixTest<TNumber>(TNumber[] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftVector.To(device);
        rightMatrix.To(device);

        var result = leftVector.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndTensor(IDevice device)
    {
        VectorTensorTest(new BFloat16[] { 7, 6 }, new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new float[] { 7, 6 }, new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new double[] { 7, 6 }, new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new byte[] { 7, 6 }, new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new sbyte[] { 7, 6 }, new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new ushort[] { 7, 6 }, new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new short[] { 7, 6 }, new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new uint[] { 7, 6 }, new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new int[] { 7, 6 }, new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new ulong[] { 7, 6 }, new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest(new long[] { 7, 6 }, new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
    }

    private static Array VectorTensorTest<TNumber>(TNumber[] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftVector.To(device);
        rightTensor.To(device);

        var result = leftVector.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndScalar(IDevice device)
    {
        MatrixScalarTest(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest(new float[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new float[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest(new double[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new double[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new byte[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new ushort[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new short[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new uint[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest(new int[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new int[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new ulong[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest(new long[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new long[,] { { 1, 3 }, { 5, 7 } });
    }

    private static Array MatrixScalarTest<TNumber>(TNumber[,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightScalar = new Scalar<TNumber>(right);
        leftMatrix.To(device);
        rightScalar.To(device);

        var result = leftMatrix.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndVector(IDevice device)
    {
        MatrixVectorTest(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new float[,] { { 2, 4 }, { 6, 8 } }, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new double[,] { { 2, 4 }, { 6, 8 } }, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new short[,] { { 2, 4 }, { 6, 8 } }, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new int[,] { { 2, 4 }, { 6, 8 } }, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 2 }, { 5, 6 } });
    }

    private static Array MatrixVectorTest<TNumber>(TNumber[,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftMatrix.To(device);
        rightVector.To(device);

        var result = leftMatrix.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 3 }, { 6, 7 } });
    }

    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftMatrix.To(device);
        rightMatrix.To(device);

        var result = leftMatrix.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndTensor(IDevice device)
    {
        MatrixTensorTest(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
    }

    private static Array MatrixTensorTest<TNumber>(TNumber[,] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftMatrix.To(device);
        rightTensor.To(device);

        var result = leftMatrix.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndScalar(IDevice device)
    {
        TensorScalarTest(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest(new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new float[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest(new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new double[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<byte>(new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new byte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<sbyte>(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<ushort>(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<short>(new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new short[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<uint>(new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new uint[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest(new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new int[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<ulong>(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest(new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new long[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
    }

    private static Array TensorScalarTest<TNumber>(TNumber[,,,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightScalar = new Scalar<TNumber>(right);
        leftTensor.To(device);
        rightScalar.To(device);

        var result = leftTensor.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndVector(IDevice device)
    {
        TensorVectorTest(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[] { 0, 1 }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[] { 0, 1 }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[] { 0, 1 }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[] { 0, 1 }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[] { 0, 1 }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[] { 0, 1 }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[] { 0, 1 }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[] { 0, 1 }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[] { 0, 1 }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[] { 0, 1 }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[] { 0, 1 }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 3, 3 } } } });
    }

    private static Array TensorVectorTest<TNumber>(TNumber[,,,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftTensor.To(device);
        rightVector.To(device);

        var result = leftTensor.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndMatrix(IDevice device)
    {
        TensorMatrixTest(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 1, 1 } } } });
    }

    private static Array TensorMatrixTest<TNumber>(TNumber[,,,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftTensor.To(device);
        rightMatrix.To(device);

        var result = leftTensor.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndTensor(IDevice device)
    {
        TensorTensorTest(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 1, 1 } } } });
    }

    private static Array TensorTensorTest<TNumber>(TNumber[,,,] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftTensor.To(device);
        rightTensor.To(device);

        var result = leftTensor.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }
}