package us.ihmc.matrixlib;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.stream.DoubleStream;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrix3x3;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.Matrix;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.ejml.ops.ConvertDMatrixStruct;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.Conversions;
import us.ihmc.commons.RandomNumbers;
import us.ihmc.euclid.matrix.Matrix3D;
import us.ihmc.euclid.tuple3D.Vector3D;

public class NativeMatrixTest
{
   private static final int maxSize = 80;
   private static final int warmumIterations = 2000;
   private static final int iterations = 100;
   private static final double epsilon = 1.0e-8;

   // Make volatile to force operation order
   private volatile long nativeTime = 0;
   private volatile long ejmlTime = 0;

   
   
   @Test
   public void testZero()
   {
      Random random = new Random(98264L);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixRMaj expected = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         NativeMatrix nativeMatrix = new NativeMatrix(expected);

         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         expected.zero();
         nativeMatrix.zero();
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }
   }

   @Test
   public void testAddElement()
   {
      Random random = new Random(98264L);

      for (int iter = 0; iter < iterations; iter++)
      {
         int rows = RandomNumbers.nextInt(random, 1, maxSize);
         int cols = RandomNumbers.nextInt(random, 1, maxSize);
         DMatrixRMaj expected = RandomMatrices_DDRM.rectangle(rows, cols, random);
         DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(rows, cols, random);
         NativeMatrix nativeMatrix = new NativeMatrix(expected);

         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         for (int i = 0; i < 10; i++)
         {
            int row = RandomNumbers.nextInt(random, 0, rows - 1);
            int col = RandomNumbers.nextInt(random, 0, cols - 1);
            double value = RandomNumbers.nextDouble(random, 10.0);
            expected.add(row, col, value);

            nativeMatrix.add(row, col, value);
            nativeMatrix.get(actual);
            MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
         }
      }
   }

   @Test
   public void testAddEquals()
   {
      Random random = new Random(98264L);

      DMatrixRMaj expected = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
      DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
      NativeMatrix nativeMatrix = new NativeMatrix(expected);

      for (int iter = 0; iter < iterations; iter++)
      {
         DMatrixRMaj b = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);

         CommonOps_DDRM.addEquals(expected, b);
         nativeMatrix.addEquals(new NativeMatrix(b));

         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         CommonOps_DDRM.addEquals(expected, scale, b);
         nativeMatrix.addEquals(scale, new NativeMatrix(b));

         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }
   }

   @Test
   public void testContainsNaN()
   {
      Random random = new Random(789896);

      for (int i = 0; i < iterations; i++)
      {
         NativeMatrix nativeMatrix = new NativeMatrix(RandomMatrices_DDRM.rectangle(maxSize, maxSize, random));
         assertFalse(nativeMatrix.containsNaN());
         nativeMatrix.set(random.nextInt(nativeMatrix.getNumRows()), random.nextInt(nativeMatrix.getNumCols()), Double.NaN);
         assertTrue(nativeMatrix.containsNaN());
      }
   }

   @Test
   public void testElementOperations()
   {
      Random random = new Random(37889);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixRMaj ejmlMatrix = RandomMatrices_DDRM.rectangle(maxSize, maxSize, -100.0, 100.0, random);
         NativeMatrix nativeMatrix = new NativeMatrix(ejmlMatrix);

         assertEquals(CommonOps_DDRM.elementMin(ejmlMatrix), nativeMatrix.min());
         assertEquals(CommonOps_DDRM.elementMax(ejmlMatrix), nativeMatrix.max());
         double expectedSum = CommonOps_DDRM.elementSum(ejmlMatrix);
         double actualSum = nativeMatrix.sum();
         assertEquals(expectedSum, actualSum, epsilon, "Error: " + (expectedSum - actualSum));
         double expectedProd = DoubleStream.of(ejmlMatrix.data).reduce(1.0, (a, b) -> a * b);
         double actualProd = nativeMatrix.prod();
         assertEquals(expectedProd, actualProd, epsilon, "Error: " + (expectedProd - actualProd));
      }
   }

   @Test
   public void testScale()
   {
      Random random = new Random(40L);

      System.out.println("Testing matrix set-and-scale with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         double alpha = RandomNumbers.nextDouble(random, 10.0);
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj B = new DMatrixRMaj(1, 1);
         CommonOps_DDRM.scale(alpha, A, B);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);

         nativeA.set(A);
         nativeB.scale(alpha, nativeA);
      }

      for (int i = 0; i < iterations; i++)
      {
         int rows = random.nextInt(maxSize) + 1;
         int cols = random.nextInt(maxSize) + 1;
         matrixSizes += (rows + cols) / 2.0;

         double alpha = RandomNumbers.nextDouble(random, 10.0);
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(rows, cols, random);
         DMatrixRMaj actual = new DMatrixRMaj(rows, cols);
         DMatrixRMaj expected = new DMatrixRMaj(rows, cols);

         NativeMatrix nativeA = new NativeMatrix(rows, cols);
         NativeMatrix nativeB = new NativeMatrix(rows, cols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.scale(alpha, nativeA);
         nativeB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.scale(alpha, A, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      System.out.println("Test A:");
      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);

      nativeTime = 0;
      ejmlTime = 0;
      matrixSizes = 0.0;

      for (int i = 0; i < iterations; i++)
      {
         int rows = random.nextInt(maxSize) + 1;
         int cols = random.nextInt(maxSize) + 1;
         matrixSizes += (rows + cols) / 2.0;

         double alpha = RandomNumbers.nextDouble(random, 10.0);
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(rows, cols, random);
         DMatrixRMaj actual = new DMatrixRMaj(rows, cols);
         DMatrixRMaj expected = new DMatrixRMaj(rows, cols);

         NativeMatrix nativeB = new NativeMatrix(rows, cols);

         nativeTime -= System.nanoTime();
         nativeB.scale(alpha, A);
         nativeB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.scale(alpha, A, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      System.out.println("Test B:");
      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");
   }

   @Test
   public void testMult()
   {
      Random random = new Random(40L);

      System.out.println("Testing matrix multiplications with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj AB = new DMatrixRMaj(maxSize, maxSize);
         CommonOps_DDRM.mult(A, B, AB);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeAB = new NativeMatrix(maxSize, maxSize);

         nativeA.set(A);
         nativeB.set(B);
         nativeAB.mult(nativeA, nativeB);
         nativeAB.get(AB);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         int bCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols + bCols) / 3.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(aCols, bCols, random);
         DMatrixRMaj actual = new DMatrixRMaj(aRows, bCols);
         DMatrixRMaj expected = new DMatrixRMaj(aRows, bCols);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);
         NativeMatrix nativeB = new NativeMatrix(aCols, bCols);
         NativeMatrix nativeAB = new NativeMatrix(aRows, bCols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAB.mult(nativeA, nativeB);
         nativeAB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.mult(A, B, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         NativeMatrix matrix = new NativeMatrix(1, 1);

         assertDoesNotThrow(() -> matrix.mult(new NativeMatrix(5, 3), new NativeMatrix(3, 7)));
         assertDoesNotThrow(() -> matrix.mult(new NativeMatrix(0, 3), new NativeMatrix(3, 7)));
         assertDoesNotThrow(() -> matrix.mult(new NativeMatrix(5, 3), new NativeMatrix(3, 0)));
         assertDoesNotThrow(() -> matrix.mult(new NativeMatrix(5, 0), new NativeMatrix(0, 7)));
         assertThrows(IllegalArgumentException.class, () -> matrix.mult(new NativeMatrix(5, 3), new NativeMatrix(8, 7)));
         assertThrows(IllegalArgumentException.class, () -> matrix.mult(new NativeMatrix(5, 13), new NativeMatrix(8, 7)));
      }
   }

   @Test
   public void testMultAdd()
   {
      Random random = new Random(40L);

      System.out.println("Testing matrix multiplications with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj AB = new DMatrixRMaj(maxSize, maxSize);
         CommonOps_DDRM.multAdd(A, B, AB);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeAB = new NativeMatrix(maxSize, maxSize);

         nativeA.set(A);
         nativeB.set(B);
         nativeAB.multAdd(nativeA, nativeB);
         nativeAB.get(AB);

         double scale = RandomNumbers.nextDouble(random, 10.0);
         CommonOps_DDRM.multAdd(scale, A, B, AB);
         nativeAB.multAdd(scale, nativeA, nativeB);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         int bCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols + bCols) / 3.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(aCols, bCols, random);
         DMatrixRMaj actual = new DMatrixRMaj(aRows, bCols);
         DMatrixRMaj expected = RandomMatrices_DDRM.rectangle(aRows, bCols, random);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);
         NativeMatrix nativeB = new NativeMatrix(aCols, bCols);
         NativeMatrix nativeAB = new NativeMatrix(aRows, bCols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAB.set(expected);
         nativeAB.multAdd(nativeA, nativeB);
         nativeAB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.multAdd(A, B, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         nativeA.set(A);
         nativeB.set(B);
         nativeAB.set(expected);
         nativeAB.multAdd(scale, nativeA, nativeB);
         nativeAB.get(actual);

         CommonOps_DDRM.multAdd(scale, A, B, expected);

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(5, 3), new NativeMatrix(3, 7)));
         assertDoesNotThrow(() -> new NativeMatrix(0, 7).multAdd(new NativeMatrix(0, 3), new NativeMatrix(3, 7)));
         assertDoesNotThrow(() -> new NativeMatrix(5, 0).multAdd(new NativeMatrix(5, 3), new NativeMatrix(3, 0)));
         assertDoesNotThrow(() -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(5, 0), new NativeMatrix(0, 7)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(5, 3), new NativeMatrix(8, 7)));
         assertThrows(expectedType, () -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(5, 13), new NativeMatrix(8, 7)));
         assertThrows(expectedType, () -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(6, 3), new NativeMatrix(3, 7)));
         assertThrows(expectedType, () -> new NativeMatrix(5, 7).multAdd(new NativeMatrix(5, 3), new NativeMatrix(3, 5)));
      }
   }

   @Test
   public void testMultQuad()
   {
      Random random = new Random(40L);

      System.out.println("Testing computing quadratic form with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj tempBA = new DMatrixRMaj(maxSize, maxSize);
         DMatrixRMaj AtBA = new DMatrixRMaj(maxSize, maxSize);
         CommonOps_DDRM.mult(B, A, tempBA);
         CommonOps_DDRM.multTransA(A, tempBA, AtBA);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeAtBA = new NativeMatrix(maxSize, maxSize);
         nativeA.set(A);
         nativeB.set(B);
         nativeAtBA.multQuad(nativeA, nativeB);
         nativeAtBA.get(AtBA);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols) / 2.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(aRows, aRows, random);
         DMatrixRMaj actual = new DMatrixRMaj(aCols, aCols);
         DMatrixRMaj expected = new DMatrixRMaj(aCols, aCols);
         DMatrixRMaj tempBA = new DMatrixRMaj(aRows, aCols);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);
         NativeMatrix nativeB = new NativeMatrix(aRows, aRows);
         NativeMatrix nativeAtBA = new NativeMatrix(aCols, aCols);

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.mult(B, A, tempBA);
         CommonOps_DDRM.multTransA(A, tempBA, expected);
         ejmlTime += System.nanoTime();

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAtBA.multQuad(nativeA, nativeB);
         nativeAtBA.get(actual);
         nativeTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeMatrix(8, 8).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(0, 3), new NativeMatrix(0, 0)));
         assertDoesNotThrow(() -> new NativeMatrix(0, 0).multQuad(new NativeMatrix(5, 0), new NativeMatrix(5, 5)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(6, 5)));
         assertThrows(expectedType, () -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 6)));
      }
   }

   @Test
   public void testMultAddQuad()
   {
      Random random = new Random(40L);

      System.out.println("Testing computing quadratic form with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj base = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj tempBA = new DMatrixRMaj(maxSize, maxSize);
         DMatrixRMaj AtBA = new DMatrixRMaj(maxSize, maxSize);
         CommonOps_DDRM.mult(B, A, tempBA);
         CommonOps_DDRM.multTransA(A, tempBA, AtBA);
         CommonOps_DDRM.addEquals(base, AtBA);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeAtBA = new NativeMatrix(base);
         nativeA.set(A);
         nativeB.set(B);
         nativeAtBA.multAddQuad(nativeA, nativeB);
         nativeAtBA.get(AtBA);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols) / 2.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(aRows, aRows, random);
         DMatrixRMaj base = RandomMatrices_DDRM.rectangle(aCols, aCols, random);
         DMatrixRMaj actual = new DMatrixRMaj(aCols, aCols);
         DMatrixRMaj expected = new DMatrixRMaj(aCols, aCols);
         DMatrixRMaj tempBA = new DMatrixRMaj(aRows, aCols);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);
         NativeMatrix nativeB = new NativeMatrix(aRows, aRows);
         NativeMatrix nativeAtBA = new NativeMatrix(base);

         ejmlTime -= System.nanoTime();
         expected.set(base);
         CommonOps_DDRM.mult(B, A, tempBA);
         CommonOps_DDRM.multAddTransA(A, tempBA, expected);
         ejmlTime += System.nanoTime();

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAtBA.multAddQuad(nativeA, nativeB);
         nativeAtBA.get(actual);
         nativeTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeMatrix(8, 8).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(0, 3), new NativeMatrix(0, 0)));
         assertDoesNotThrow(() -> new NativeMatrix(0, 0).multQuad(new NativeMatrix(5, 0), new NativeMatrix(5, 5)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(6, 5)));
         assertThrows(expectedType, () -> new NativeMatrix(3, 3).multQuad(new NativeMatrix(5, 3), new NativeMatrix(5, 6)));
      }
   }

   @Test
   public void testInvert()
   {
      Random random = new Random(40L);

      System.out.println("Testing inverting with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolverDense<DMatrixRMaj> solver = LinearSolverFactory_DDRM.lu(maxSize);

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, -100.0, 100.0, random);
         DMatrixRMaj B = new DMatrixRMaj(maxSize, maxSize);
         solver.setA(A);
         solver.invert(B);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativeB = new NativeMatrix(maxSize, maxSize);
         nativeA.set(A);
         nativeB.invert(nativeA);
         nativeB.get(B);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aRows, -100.0, 100.0, random);
         DMatrixRMaj nativeResult = new DMatrixRMaj(aRows, aRows);
         DMatrixRMaj ejmlResult = new DMatrixRMaj(aRows, aRows);

         NativeMatrix nativeA = new NativeMatrix(aRows, aRows);
         NativeMatrix nativeB = new NativeMatrix(aRows, aRows);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.invert(nativeA);
         nativeB.get(nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         solver.setA(A);
         solver.invert(ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(ejmlResult, nativeResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Text exception
         NativeMatrix nativeMatrix = new NativeMatrix(RandomMatrices_DDRM.rectangle(random.nextInt(maxSize), random.nextInt(maxSize), random));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.invert(nativeMatrix));
      }
   }

   @Test
   public void testRemoveRow()
   {
      Random random = new Random(40L);

      System.out.println("Testing removing row with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, -100.0, 100.0, random);

         MatrixTools.removeRow(A, 3);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         nativeA.set(A);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, -100.0, 100.0, random);
         DMatrixRMaj nativeResult = new DMatrixRMaj(aRows, aCols);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);

         int rowToRemove = aRows == 1 ? 0 : random.nextInt(aRows - 1);

         nativeA.set(A);
         nativeTime -= System.nanoTime();
         nativeA.removeRow(rowToRemove);
         nativeTime += System.nanoTime();
         nativeA.get(nativeResult);

         ejmlTime -= System.nanoTime();
         MatrixTools.removeRow(A, rowToRemove);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(A, nativeResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         NativeMatrix nativeMatrix = new NativeMatrix(20, 20);
         assertDoesNotThrow(() -> nativeMatrix.removeRow(0));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> nativeMatrix.removeRow(-1));
         assertThrows(expectedType, () -> nativeMatrix.removeRow(nativeMatrix.getNumRows()));
      }
   }

   @Test
   public void testRemoveColumn()
   {
      Random random = new Random(40L);

      System.out.println("Testing removing column with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, -100.0, 100.0, random);

         MatrixTools.removeColumn(A, 3);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         nativeA.set(A);
         nativeA.removeColumn(3);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aCols, -100.0, 100.0, random);
         DMatrixRMaj nativeResult = new DMatrixRMaj(aRows, aCols);

         NativeMatrix nativeA = new NativeMatrix(aRows, aCols);

         int colToRemove = aCols == 1 ? 0 : random.nextInt(aCols - 1);

         nativeA.set(A);
         nativeTime -= System.nanoTime();
         nativeA.removeColumn(colToRemove);
         nativeTime += System.nanoTime();
         nativeA.get(nativeResult);

         ejmlTime -= System.nanoTime();
         MatrixTools.removeColumn(A, colToRemove);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(A, nativeResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         NativeMatrix nativeMatrix = new NativeMatrix(20, 20);
         assertDoesNotThrow(() -> nativeMatrix.removeColumn(0));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> nativeMatrix.removeColumn(-1));
         assertThrows(expectedType, () -> nativeMatrix.removeColumn(nativeMatrix.getNumRows()));
      }
   }

   @Test
   public void testSolve()
   {
      Random random = new Random(40L);

      System.out.println("Testing solving linear equations with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolverDense<DMatrixRMaj> solver = LinearSolverFactory_DDRM.lu(maxSize);

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj x = RandomMatrices_DDRM.rectangle(maxSize, 1, random);
         DMatrixRMaj b = new DMatrixRMaj(maxSize, 1);
         CommonOps_DDRM.mult(A, x, b);
         solver.setA(A);
         solver.solve(b, x);

         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);
         NativeMatrix nativex = new NativeMatrix(maxSize, 1);
         NativeMatrix nativeb = new NativeMatrix(maxSize, 1);
         nativeA.set(A);
         nativex.set(x);
         nativeb.set(b);
         nativex.solve(nativeA, nativeb);
         nativex.get(x);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, aRows, random);
         DMatrixRMaj x = RandomMatrices_DDRM.rectangle(aRows, 1, random);
         DMatrixRMaj b = new DMatrixRMaj(aRows, 1);
         CommonOps_DDRM.mult(A, x, b);

         DMatrixRMaj nativeResult = new DMatrixRMaj(aRows, 1);
         DMatrixRMaj ejmlResult = new DMatrixRMaj(aRows, 1);

         NativeMatrix nativeA = new NativeMatrix(aRows, aRows);
         NativeMatrix nativex = new NativeMatrix(aRows, 1);
         NativeMatrix nativeb = new NativeMatrix(aRows, 1);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeb.set(b);
         nativex.solve(nativeA, nativeb);
         nativex.get(nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         solver.setA(A);
         solver.solve(b, ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(x, nativeResult, epsilon);
         MatrixTestTools.assertMatrixEquals(x, ejmlResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeMatrix(15, 11).solve(new NativeMatrix(5, 5), new NativeMatrix(5, 1)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeMatrix(15, 11).solve(new NativeMatrix(5, 5), new NativeMatrix(5, 2)));
         assertThrows(expectedType, () -> new NativeMatrix(15, 11).solve(new NativeMatrix(5, 5), new NativeMatrix(4, 1)));
         assertThrows(expectedType, () -> new NativeMatrix(15, 11).solve(new NativeMatrix(5, 4), new NativeMatrix(5, 1)));
         assertThrows(expectedType, () -> new NativeMatrix(15, 11).solve(new NativeMatrix(6, 5), new NativeMatrix(5, 1)));
      }
   }

   private static void printTimings(long nativeTotalTime, long ejmlTotalTime, double summedMatrixSizes, int iterations)
   {
      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTotalTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTotalTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + summedMatrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTotalTime / ejmlTotalTime + "% of EJML time.");
   }

   @Test
   public void testMultAddBlock()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         NativeMatrix randomMatrixA = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, taskSize, -50.0, 50.0, random));
         NativeMatrix randomMatrixB = new NativeMatrix(RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random));

         NativeMatrix solution = new NativeMatrix(RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random));

         NativeMatrix expectedSolution = new NativeMatrix(solution);

         NativeMatrix temp = new NativeMatrix(rows, cols);
         temp.mult(randomMatrixA, randomMatrixB);

         expectedSolution.addBlock(temp, rowStart, colStart, 0, 0, rows, cols, 1.0);

         solution.multAddBlock(randomMatrixA, randomMatrixB, rowStart, colStart);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));

         double scale = RandomNumbers.nextDouble(random, 10.0);

         expectedSolution.addBlock(temp, rowStart, colStart, 0, 0, rows, cols, scale);

         solution.multAddBlock(scale, randomMatrixA, randomMatrixB, rowStart, colStart);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }
   }

   @Test
   public void testMultAddBlockTransA()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         NativeMatrix randomMatrixA = new NativeMatrix(RandomMatrices_DDRM.rectangle(taskSize, rows, -50.0, 50.0, random));
         NativeMatrix randomMatrixB = new NativeMatrix(RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random));

         NativeMatrix solution = new NativeMatrix(RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random));

         NativeMatrix expectedSolution = new NativeMatrix(solution);

         NativeMatrix temp = new NativeMatrix(rows, cols);
         temp.multTransA(randomMatrixA, randomMatrixB);

         expectedSolution.addBlock(temp, rowStart, colStart, 0, 0, rows, cols, 1.0);

         solution.multAddBlockTransA(randomMatrixA, randomMatrixB, rowStart, colStart);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));

         double scale = RandomNumbers.nextDouble(random, 10.0);

         expectedSolution.addBlock(temp, rowStart, colStart, 0, 0, rows, cols, scale);

         solution.multAddBlockTransA(scale, randomMatrixA, randomMatrixB, rowStart, colStart);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }
   }
   
   @Test
   public void testAddBlock()
   {
      Random random = new Random(349754);
      
      for (int i = 0; i < iterations; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         NativeMatrix randomMatrixA = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, cols, -50.0, 50.0, random));

         NativeMatrix solution = new NativeMatrix(RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random));

         NativeMatrix expectedSolution = new NativeMatrix(solution);


         double scale = RandomNumbers.nextDouble(random, 10.0);
         expectedSolution.addBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols, scale);

         solution.addBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols, scale);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }

      { // Test exceptions
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int destRowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int destColStart = RandomNumbers.nextInt(random, 0, fullCols - cols);
         int srcStartRow = 0;
         int srcStartColumn = 0;

         NativeMatrix expectedSolution = new NativeMatrix(fullRows, fullCols);
         NativeMatrix block = new NativeMatrix(rows, cols);

         assertDoesNotThrow(() -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols, Double.NaN));
         assertTrue(expectedSolution.containsNaN());
         assertDoesNotThrow(() -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols, 1.0));

         Class<IllegalArgumentException> exceptionType = IllegalArgumentException.class;
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, -1, destColStart, srcStartRow, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, -1, srcStartRow, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, -1, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, -1, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, -1, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, -1, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, fullRows - rows + 1, destColStart, srcStartRow, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, fullCols - cols + 1, srcStartRow, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow + 1, srcStartColumn, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn + 1, rows, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows + 1, cols, 1.0));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols + 1, 1.0));
      }
   }

   @Test
   public void testAddBlockNoScale()
   {
      Random random = new Random(349754);
      
      for (int i = 0; i < iterations; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         NativeMatrix randomMatrixA = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, cols, -50.0, 50.0, random));

         NativeMatrix solution = new NativeMatrix(RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random));

         NativeMatrix expectedSolution = new NativeMatrix(solution);


         expectedSolution.addBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols, 1.0);

         solution.addBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }

      { // Test exceptions
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int destRowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int destColStart = RandomNumbers.nextInt(random, 0, fullCols - cols);
         int srcStartRow = 0;
         int srcStartColumn = 0;

         NativeMatrix expectedSolution = new NativeMatrix(fullRows, fullCols);
         NativeMatrix block = new NativeMatrix(rows, cols);

         assertDoesNotThrow(() -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols));

         Class<IllegalArgumentException> exceptionType = IllegalArgumentException.class;
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, -1, destColStart, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, -1, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, -1, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, -1, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, -1, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, -1));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, fullRows - rows + 1, destColStart, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, fullCols - cols + 1, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow + 1, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn + 1, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows + 1, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols + 1));
      }
   }
   
   @Test
   public void testSubtractBlock()
   {
      Random random = new Random(349754);
      
      for (int i = 0; i < iterations; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         NativeMatrix randomMatrixA = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, cols, -50.0, 50.0, random));

         NativeMatrix solution = new NativeMatrix(RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random));

         NativeMatrix expectedSolution = new NativeMatrix(solution);


         expectedSolution.addBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols, -1.0);

         solution.subtractBlock(randomMatrixA, rowStart, colStart, 0, 0, rows, cols);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }
      
      { // Test exceptions
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);
         
         int destRowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int destColStart = RandomNumbers.nextInt(random, 0, fullCols - cols);
         int srcStartRow = 0;
         int srcStartColumn = 0;
         
         NativeMatrix expectedSolution = new NativeMatrix(fullRows, fullCols);
         NativeMatrix block = new NativeMatrix(rows, cols);
         
         assertDoesNotThrow(() -> expectedSolution.subtractBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols));
         
         Class<IllegalArgumentException> exceptionType = IllegalArgumentException.class;
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, -1, destColStart, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, -1, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, -1, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, -1, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, -1, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, -1));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, fullRows - rows + 1, destColStart, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, fullCols - cols + 1, srcStartRow, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow + 1, srcStartColumn, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn + 1, rows, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows + 1, cols));
         assertThrows(exceptionType, () -> expectedSolution.addBlock(block, destRowStart, destColStart, srcStartRow, srcStartColumn, rows, cols + 1));
      }
   }

   @Test
   public void testMultTransA()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Arows, Bcols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Acols, Bcols);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.multTransA(A, B, solution);
         nativeSolution.multTransA(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Acols, Bcols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         CommonOps_DDRM.multTransA(scale, A, B, solution);
         nativeSolution.multTransA(scale, nativeA, nativeB);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);
      }
   }

   @Test
   public void testMultAddTransA()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Arows, Bcols, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(Acols, Bcols, random);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(solution);

         CommonOps_DDRM.multAddTransA(A, B, solution);
         nativeSolution.multAddTransA(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Acols, Bcols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         CommonOps_DDRM.multAddTransA(scale, A, B, solution);
         nativeSolution.multAddTransA(scale, nativeA, nativeB);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);
      }
   }

   @Test
   public void testMultTransB()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Acols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Arows, Brows);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.multTransB(A, B, solution);
         nativeSolution.multTransB(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Arows, Brows);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         CommonOps_DDRM.multTransB(scale, A, B, solution);
         nativeSolution.multTransB(scale, nativeA, nativeB);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);
      }
   }

   @Test
   public void testMultAddTransB()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Acols, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(Arows, Brows, random);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(solution);

         CommonOps_DDRM.multAddTransB(A, B, solution);
         nativeSolution.multAddTransB(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Arows, Brows);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);

         double scale = RandomNumbers.nextDouble(random, 10.0);

         CommonOps_DDRM.multAddTransB(scale, A, B, solution);
         nativeSolution.multAddTransB(scale, nativeA, nativeB);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-10);
      }
   }

   @Test
   public void testInsert()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);

         CommonOps_DDRM.extract(B, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insert(nativeB, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset);

         DMatrixRMaj nativeADMatrix = new DMatrixRMaj(A.getNumRows(), A.getNumCols());
         nativeA.get(nativeADMatrix);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);

         CommonOps_DDRM.insert(B, A, ArowOffset, AcolOffset);
         nativeA.insert(nativeB, ArowOffset, AcolOffset);
         nativeA.get(nativeADMatrix);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
      }

      { // Test exceptions
         int Arows = 10;
         int Acols = 10;
         int Brows = 5;
         int Bcols = 5;

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int blockRows = Brows - BrowOffset;
         int blockCols = Bcols - BcolOffset;

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(Brows, Bcols, random));

         assertDoesNotThrow(() -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> A.insert(B, -1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, -1, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, -1, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, -1, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, -1, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, -1));

         assertThrows(expectedType, () -> A.insert(B, Brows + 1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows + 1, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, Bcols + 1, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols + 1, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, Arows - blockRows + 1, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, Acols - blockCols + 1));
      }
   }
   
   @Test
   public void testInsertScaled()
   {
      Random random = new Random(124L);
      
      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);
         
         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);
         
         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);
         
         double scale = RandomNumbers.nextDouble(random, 10);
         
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);
         
         DMatrixRMaj BScaled = new DMatrixRMaj(B);
         CommonOps_DDRM.scale(scale, BScaled);
         
         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         
         CommonOps_DDRM.extract(BScaled, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insertScaled(nativeB, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, scale);
         
         DMatrixRMaj nativeADMatrix = new DMatrixRMaj(A.getNumRows(), A.getNumCols());
         nativeA.get(nativeADMatrix);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
         
         CommonOps_DDRM.insert(BScaled, A, ArowOffset, AcolOffset);
         nativeA.insertScaled(nativeB, ArowOffset, AcolOffset, scale);
         nativeA.get(nativeADMatrix);
         
         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
      }
      
      { // Test exceptions
         int Arows = 10;
         int Acols = 10;
         int Brows = 5;
         int Bcols = 5;
         
         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);
         
         int blockRows = Brows - BrowOffset;
         int blockCols = Bcols - BcolOffset;
         
         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);
         
         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(Brows, Bcols, random));

         assertDoesNotThrow(() -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> A.insertScaled(B, -1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, -1, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, -1, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, -1, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, -1, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, -1, 1.0));

         assertThrows(expectedType, () -> A.insertScaled(B, Brows + 1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows + 1, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, Bcols + 1, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols + 1, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, Arows - blockRows + 1, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, Acols - blockCols + 1, 1.0));
      }
   }

   @Test
   public void testEJMLInsert()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);

         NativeMatrix nativeA = new NativeMatrix(A);

         CommonOps_DDRM.extract(B, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset);

         DMatrixRMaj nativeADMatrix = new DMatrixRMaj(A.getNumRows(), A.getNumCols());
         nativeA.get(nativeADMatrix);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);

         CommonOps_DDRM.insert(B, A, ArowOffset, AcolOffset);
         nativeA.insert(B, ArowOffset, AcolOffset);
         nativeA.get(nativeADMatrix);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
      }

      { // Test exceptions
         int Arows = 10;
         int Acols = 10;
         int Brows = 5;
         int Bcols = 5;

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int blockRows = Brows - BrowOffset;
         int blockCols = Bcols - BcolOffset;

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);

         assertDoesNotThrow(() -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> A.insert(B, -1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, -1, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, -1, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, -1, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, -1, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, -1));

         assertThrows(expectedType, () -> A.insert(B, Brows + 1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows + 1, BcolOffset, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, Bcols + 1, Bcols, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols + 1, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, Arows - blockRows + 1, AcolOffset));
         assertThrows(expectedType, () -> A.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, Acols - blockCols + 1));
      }
   }
   
   @Test
   public void testEJMLInsertScaled()
   {
      Random random = new Random(124L);
      
      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);
         
         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);
         
         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);
         
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);
         
         
         double scale = RandomNumbers.nextDouble(random, 10.0);
         DMatrixRMaj BScaled = new DMatrixRMaj(B);
         CommonOps_DDRM.scale(scale, BScaled);
         
         NativeMatrix nativeA = new NativeMatrix(A);
         
         CommonOps_DDRM.extract(BScaled, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, scale);
         
         DMatrixRMaj nativeADMatrix = new DMatrixRMaj(A.getNumRows(), A.getNumCols());
         nativeA.get(nativeADMatrix);
         
         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
         
         CommonOps_DDRM.insert(BScaled, A, ArowOffset, AcolOffset);
         nativeA.insertScaled(B, ArowOffset, AcolOffset, scale);
         nativeA.get(nativeADMatrix);
         
         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
      }
      
      { // Test exceptions
         int Arows = 10;
         int Acols = 10;
         int Brows = 5;
         int Bcols = 5;
         
         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);
         
         int blockRows = Brows - BrowOffset;
         int blockCols = Bcols - BcolOffset;
         
         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);
         
         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);
         
         assertDoesNotThrow(() -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> A.insertScaled(B, -1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, -1, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, -1, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, -1, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, -1, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, -1, 1.0));

         assertThrows(expectedType, () -> A.insertScaled(B, Brows + 1, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows + 1, BcolOffset, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, Bcols + 1, Bcols, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols + 1, ArowOffset, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, Arows - blockRows + 1, AcolOffset, 1.0));
         assertThrows(expectedType, () -> A.insertScaled(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, Acols - blockCols + 1, 1.0));
      }
   }

   @Test
   public void testEJMLExtract()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Bcols, random);
         DMatrixRMaj nativeADMatrix = new DMatrixRMaj(A);

         NativeMatrix nativeB = new NativeMatrix(B);

         CommonOps_DDRM.extract(B, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeB.extract(BrowOffset, Brows, BcolOffset, Bcols, nativeADMatrix, ArowOffset, AcolOffset);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);

         CommonOps_DDRM.insert(B, A, ArowOffset, AcolOffset);
         nativeB.extract(nativeADMatrix, ArowOffset, AcolOffset);

         MatrixTestTools.assertMatrixEquals(A, nativeADMatrix, 1.0e-10);
      }

      { // Test exceptions
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Brows = RandomNumbers.nextInt(random, 1, Arows);
         int Bcols = RandomNumbers.nextInt(random, 1, Acols);

         int BrowOffset = RandomNumbers.nextInt(random, 0, Brows);
         int BcolOffset = RandomNumbers.nextInt(random, 0, Bcols);

         int ArowOffset = RandomNumbers.nextInt(random, 0, Arows - Brows);
         int AcolOffset = RandomNumbers.nextInt(random, 0, Acols - Bcols);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(Brows, Bcols, random));

         assertDoesNotThrow(() -> B.extract(BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> B.extract(-1, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, -1, BcolOffset, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, -1, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, -1, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, Bcols, A, -1, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, -1));

         assertThrows(expectedType, () -> B.extract(Brows + 1, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows + 1, BcolOffset, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, Bcols + 1, Bcols, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, Bcols + 1, A, ArowOffset, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, Bcols, A, Arows - (Brows - BrowOffset) + 1, AcolOffset));
         assertThrows(expectedType, () -> B.extract(BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, Acols - (Bcols - BcolOffset) + 1));
      }
   }
   
   @Test
   public void testInsertMatrix3D()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         Matrix3D matrix = new Matrix3D();
         matrix.set(RandomMatrices_DDRM.rectangle(9, 9, random));
         
         
         DMatrixRMaj expected = new DMatrixRMaj(100, 100);
         NativeMatrix actual = new NativeMatrix(100, 100);
         
         int Arows = RandomNumbers.nextInt(random, 0, 90);
         int Acols = RandomNumbers.nextInt(random, 0, 90);

         
         matrix.get(Arows, Acols, expected);
         
         actual.insert(matrix, Arows, Acols);
         
         
         MatrixTestTools.assertMatrixEquals(expected, actual, 1e-10);
      }
   }
   @Test
   public void testInsertTuple()
   {
      Random random = new Random(124L);
      
      for (int i = 0; i < iterations; i++)
      {
         
         Vector3D tuple = new Vector3D();
         tuple.set(RandomMatrices_DDRM.rectangle(3, 1, random));
         
         
         DMatrixRMaj expected = new DMatrixRMaj(100, 100);
         NativeMatrix actual = new NativeMatrix(100, 100);
         NativeMatrix actual2 = new NativeMatrix(100, 100);
         
         int Arows = RandomNumbers.nextInt(random, 0, 90);
         int Acols = RandomNumbers.nextInt(random, 0, 90);
         
         
         tuple.get(Arows, Acols, expected);
         
         actual.insertTupleRow(tuple, Arows, Acols);
         actual2.insertTupleRow(Arows, Acols, tuple.getX(), tuple.getY(), tuple.getZ());
         
         
         MatrixTestTools.assertMatrixEquals(expected, actual, 1e-10);
         MatrixTestTools.assertMatrixEquals(expected, actual2, 1e-10);
      }
   }
   
   @Test
   public void testInsertScaledMatrix3D()
   {
      Random random = new Random(124L);
      
      for (int i = 0; i < iterations; i++)
      {
         Matrix3D matrix = new Matrix3D();
         matrix.set(RandomMatrices_DDRM.rectangle(9, 9, random));
         
         double scale = RandomNumbers.nextDouble(random, 10.0);
         
         DMatrixRMaj expected = new DMatrixRMaj(100, 100);
         NativeMatrix actual = new NativeMatrix(100, 100);
         
         int Arows = RandomNumbers.nextInt(random, 0, 90);
         int Acols = RandomNumbers.nextInt(random, 0, 90);
         
         
         matrix.get(Arows, Acols, expected);
         CommonOps_DDRM.scale(scale, expected);
         
         actual.insertScaled(matrix, Arows, Acols, scale);
         
         
         MatrixTestTools.assertMatrixEquals(expected, actual, 1e-10);
      }
   }

   @Test
   public void testMultScale()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         double scale = RandomNumbers.nextDouble(random, 10000.0);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Acols, Bcols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Arows, Bcols);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.mult(scale, A, B, solution);
         nativeSolution.mult(scale, nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Arows, Bcols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }
   }

   @Test
   public void testAdd()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Arows, Acols);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.add(A, B, solution);
         nativeSolution.add(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Arows, Acols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }

      for (int i = 0; i < iterations; i++)
      { // Test with one of the arguments being the same instance as the owner on which the operation is performed.
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, cols, random));
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(rows, cols, random));
         NativeMatrix expected = new NativeMatrix(0, 0);
         NativeMatrix actual = new NativeMatrix(0, 0);

         expected.add(A, B);
         DMatrixRMaj ejmlExpected = new DMatrixRMaj(rows, cols);
         expected.get(ejmlExpected);

         actual.set(B);
         actual.add(A, actual);

         DMatrixRMaj ejmlActual = new DMatrixRMaj(rows, cols);
         actual.get(ejmlActual);

         MatrixTestTools.assertMatrixEquals(expected, actual, 1.0e-7);

         actual.set(A);
         actual.add(actual, B);

         actual.get(ejmlActual);

         MatrixTestTools.assertMatrixEquals(expected, actual, 1.0e-7);

         B.set(A);
         expected.add(A, B);
         actual.set(A);
         actual.add(actual, actual);

         actual.get(ejmlActual);

         MatrixTestTools.assertMatrixEquals(expected, actual, 1.0e-7);

      }

      for (int i = 0; i < iterations; i++)
      { // Test exceptions
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         int Brows = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         if (random.nextInt(5) == 0)
            Arows = Brows;
         if (random.nextInt(5) == 0)
            Acols = Bcols;

         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(Brows, Bcols, random));
         NativeMatrix nativeMatrix = new NativeMatrix(1, 1);

         if (Arows != Brows || Acols != Bcols)
            assertThrows(IllegalArgumentException.class, () -> nativeMatrix.add(A, B));
         else
            assertDoesNotThrow(() -> nativeMatrix.add(A, B));
      }
   }

   @Test
   public void testSubtract()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Arows, Acols);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeB = new NativeMatrix(B);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.subtract(A, B, solution);
         nativeSolution.subtract(nativeA, nativeB);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Arows, Acols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }

      for (int i = 0; i < iterations; i++)
      { // Test exceptions
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         int Brows = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         if (random.nextInt(5) == 0)
            Arows = Brows;
         if (random.nextInt(5) == 0)
            Acols = Bcols;

         NativeMatrix A = new NativeMatrix(RandomMatrices_DDRM.rectangle(Arows, Acols, random));
         NativeMatrix B = new NativeMatrix(RandomMatrices_DDRM.rectangle(Brows, Bcols, random));
         NativeMatrix nativeMatrix = new NativeMatrix(1, 1);

         if (Arows != Brows || Acols != Bcols)
            assertThrows(IllegalArgumentException.class, () -> nativeMatrix.add(A, B));
         else
            assertDoesNotThrow(() -> nativeMatrix.subtract(A, B));
      }
   }

   @Test
   public void testTranspose()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Acols, Arows);

         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeSolution = new NativeMatrix(0, 0);

         CommonOps_DDRM.transpose(A, solution);
         nativeSolution.transpose(nativeA);

         DMatrixRMaj nativeSolutionDMatrix = new DMatrixRMaj(Acols, Arows);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }

      { // Text exception
         NativeMatrix nativeMatrix = new NativeMatrix(RandomMatrices_DDRM.rectangle(random.nextInt(maxSize), random.nextInt(maxSize), random));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.transpose(nativeMatrix));
      }
   }

   @Test
   public void testGetElement()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);

         NativeMatrix nativeA = new NativeMatrix(A);

         for (int j = 0; j < iterations; j++)
         {
            int row = RandomNumbers.nextInt(random, 0, Arows - 1);
            int col = RandomNumbers.nextInt(random, 0, Acols - 1);

            assertEquals(A.get(row, col), nativeA.get(row, col), 1e-10);
         }
      }

      { // Test exceptions
         NativeMatrix nativeMatrix = new NativeMatrix(RandomMatrices_DDRM.rectangle(20, 30, random));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.get(-1, 0));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.get(0, -1));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.get(nativeMatrix.getNumRows(), 0));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.get(0, nativeMatrix.getNumCols()));
      }
   }

   @Test
   public void testSetElement()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         NativeMatrix nativeA = new NativeMatrix(Arows, Acols);

         for (int j = 0; j < iterations; j++)
         {

            int row = RandomNumbers.nextInt(random, 0, Arows - 1);
            int col = RandomNumbers.nextInt(random, 0, Acols - 1);

            double next = RandomNumbers.nextDouble(random, 10000.0);
            nativeA.set(row, col, next);

            assertEquals(next, nativeA.get(row, col), 1e-10);
         }
      }

      for (int i = 0; i < iterations; i++)
      { // Test exceptions
         NativeMatrix nativeMatrix = new NativeMatrix(RandomMatrices_DDRM.rectangle(20, 30, random));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.set(-1, 0, 0));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.set(0, -1, 0));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.set(nativeMatrix.getNumRows(), 0, 0));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.set(0, nativeMatrix.getNumCols(), 0));
      }
   }

   @Test
   public void testSetMatrix()
   {
      Random random = new Random(124L);
      nativeTime = 0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         NativeMatrix nativeA = new NativeMatrix(maxSize, maxSize);

         nativeA.set(A);
      }

      double matrixSizes = 0;

      for (int i = 0; i < iterations; i++)
      { // Test set(DMatrixRMaj matrix)
         int Arows = RandomNumbers.nextInt(random, 1, maxSize);
         int Acols = RandomNumbers.nextInt(random, 1, maxSize);

         matrixSizes += (Arows + Acols) / 2.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         NativeMatrix nativeA = new NativeMatrix(Arows, Acols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeTime += System.nanoTime();

         for (int r = 0; r < Arows; r++)
         {
            for (int c = 0; c < Acols; c++)
            {
               assertEquals(A.get(r, c), nativeA.get(r, c), 1e-10);
            }
         }
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);

      for (int i = 0; i < iterations; i++)
      { // Test set(Matrix original)
         DMatrixRMaj expected = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);
         DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(maxSize, maxSize, random);

         NativeMatrix nativeMatrix = new NativeMatrix(random.nextInt(20), random.nextInt(20));

         nativeMatrix.set((Matrix) expected);
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         nativeMatrix.set(RandomMatrices_DDRM.rectangle(random.nextInt(20), random.nextInt(20), random));
         nativeMatrix.set((Matrix) new NativeMatrix(expected));
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         nativeMatrix.set(RandomMatrices_DDRM.rectangle(random.nextInt(20), random.nextInt(20), random));
         assertThrows(UnsupportedOperationException.class,
                      () -> nativeMatrix.set((Matrix) ConvertDMatrixStruct.convert(RandomMatrices_DDRM.rectangle(3, 3, random), new DMatrix3x3())));
      }
   }

   @Test
   public void testGetMatrix()
   {

      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         NativeMatrix nativeA = new NativeMatrix(Arows, Acols);

         nativeA.set(A);

         DMatrixRMaj B = new DMatrixRMaj(Arows, Acols);
         MatrixTestTools.assertMatrixEqualsZero(B, 1e-10);

         nativeA.get(B);

         MatrixTestTools.assertMatrixEquals(A, B, 1e-10);
      }
   }

   @Test
   public void testSize()
   {
      NativeMatrix nativeA = new NativeMatrix(0, 0);

      assertEquals(0, nativeA.getNumRows());
      assertEquals(0, nativeA.getNumCols());

      for (int r = 0; r < 10; r++)
      {
         for (int c = 0; c < 10; c++)
         {
            nativeA.reshape(r, c);

            assertEquals(r, nativeA.getNumRows());
            assertEquals(c, nativeA.getNumCols());
         }
      }
   }
   
   
   @Test
   public void testFillDiagonal()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 100, 200);
         int Acols = RandomNumbers.nextInt(random, 100, 200);
         
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         NativeMatrix nativeA = new NativeMatrix(A);

         
         int size = RandomNumbers.nextInt(random, 1, 49);
         int startRow = RandomNumbers.nextInt(random, 0, 49);
         int startCol = RandomNumbers.nextInt(random, 0, 49);
         
         
         double diagValue = RandomNumbers.nextDouble(random, 1000);
         MatrixTestTools.setDiagonal(A, startRow, startCol, size, diagValue);
         nativeA.fillDiagonal(startRow, startCol, size, diagValue);
         
         MatrixTestTools.assertMatrixEquals(A, nativeA, 1e-10);

      }
   }
   
   @Test
   public void testFillBlock()
   {
      Random random = new Random(124L);
      
      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 100, 200);
         int Acols = RandomNumbers.nextInt(random, 100, 200);
         
         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(Arows, Acols, random);
         NativeMatrix nativeA = new NativeMatrix(A);
         
         
         int numRows = RandomNumbers.nextInt(random, 1, 49);
         int numCols = RandomNumbers.nextInt(random, 1, 49);
         int startRow = RandomNumbers.nextInt(random, 0, 49);
         int startCol = RandomNumbers.nextInt(random, 0, 49);
         
         
         double diagValue = RandomNumbers.nextDouble(random, 1000);
         
         for(int r = startRow; r < startRow + numRows; r++)
         {            
            for(int c = startCol; c < startCol + numCols; c++)
            {
               A.set(r, c, diagValue);
            }
         }
         
         nativeA.fillBlock(startRow, startCol, numRows, numCols, diagValue);
         
         MatrixTestTools.assertMatrixEquals(A, nativeA, 1e-10);
         
      }
   }
   
}
