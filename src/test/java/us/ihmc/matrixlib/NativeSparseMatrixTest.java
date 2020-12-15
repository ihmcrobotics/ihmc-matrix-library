package us.ihmc.matrixlib;

import jdk.nashorn.internal.ir.annotations.Ignore;
import org.ejml.data.*;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.ejml.interfaces.linsol.LinearSolverSparse;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.sparse.FillReducing;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.ejml.sparse.csc.RandomMatrices_DSCC;
import org.ejml.sparse.csc.factory.LinearSolverFactory_DSCC;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import us.ihmc.commons.Conversions;
import us.ihmc.commons.RandomNumbers;

import java.util.Random;
import java.util.stream.DoubleStream;

import static org.junit.jupiter.api.Assertions.*;

public class NativeSparseMatrixTest
{
   private static final int maxSize = 4;
   private static final int sparsity = 3 * 3;
   private static final int warmumIterations = 2000;
   private static final int iterations = 2000;
   private static final double epsilon = 1.0e-8;

   // Make volatile to force operation order
   private volatile long nativeTime = 0;
   private volatile long ejmlTime = 0;

   @Test
   public void testCreation()
   {
      NativeSparseMatrix matrix = new NativeSparseMatrix(maxSize, maxSize);
      double value = 0.5;
      matrix.set(2, 1, value);
      assertEquals(value, matrix.get(2, 1));
   }

   @Test
   public void testSetFromIndex()
   {
      Random random = new Random(98264L);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(maxSize, maxSize);

         for (int row = 0; row < maxSize; row++)
         {
            for (int col = 0; col < maxSize; col++)
            {
               assertEquals(0.0, nativeMatrix.get(row, col), epsilon);

               nativeMatrix.set(row, col, expected.get(row, col));

               assertEquals(expected.get(row, col), nativeMatrix.get(row, col), epsilon);
            }
         }
      }
   }

   @Test
   public void testSetMatrix1()
   {
      Random random = new Random(98264L);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(maxSize, maxSize);

         for (int row = 0; row < maxSize; row++)
         {
            for (int col = 0; col < maxSize; col++)
            {
               assertEquals(0.0, nativeMatrix.get(row, col), epsilon);
            }
         }

         nativeMatrix.set(expected);

         for (int row = 0; row < maxSize; row++)
         {
            for (int col = 0; col < maxSize; col++)
            {
               assertEquals(expected.get(row, col), nativeMatrix.get(row, col), epsilon);
            }
         }
      }

   }

   @Test
   public void testConstruction()
   {
      Random random = new Random(98264L);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC actual = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);

         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(expected);

         for (int row = 0; row < maxSize; row++)
         {
            for (int col = 0; col < maxSize; col++)
            {
               assertEquals(expected.get(row, col), nativeMatrix.get(row, col), epsilon);
            }
         }

         // redo it, because getting the non-zero values grows the internal structure
         nativeMatrix = new NativeSparseMatrix(expected);

         //
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }
   }

   @Test
   public void testZero()
   {
      Random random = new Random(98264L);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC actual = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(expected);

         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         expected.zero();
         nativeMatrix.zero();

         for (int row = 0; row < maxSize; row++)
         {
            for (int col = 0; col < maxSize; col++)
            {
               assertEquals(0.0, nativeMatrix.get(row, col), epsilon);
            }
         }

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
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random));
         assertFalse(nativeMatrix.containsNaN());
         int row = random.nextInt(nativeMatrix.getNumRows());
         int col = random.nextInt(nativeMatrix.getNumCols());
         nativeMatrix.set(row, col, Double.NaN);
         assertTrue(nativeMatrix.containsNaN());
      }
   }

   /*
   @Test
   public void testElementOperations()
   {
      Random random = new Random(37889);

      for (int i = 0; i < iterations; i++)
      {
         DMatrixSparseCSC ejmlMatrix = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, -100.0, 100.0, random);
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(ejmlMatrix);

         assertEquals(CommonOps_DSCC.elementMin(ejmlMatrix), nativeMatrix.min());
         assertEquals(CommonOps_DSCC.elementMax(ejmlMatrix), nativeMatrix.max());
         double expectedSum = CommonOps_DSCC.elementSum(ejmlMatrix);
         double actualSum = nativeMatrix.sum();
         assertEquals(expectedSum, actualSum, epsilon, "Error: " + (expectedSum - actualSum));
         double expectedProd = DoubleStream.of(ejmlMatrix.data).reduce(1.0, (a, b) -> a * b);
         double actualProd = nativeMatrix.prod();
         assertEquals(expectedProd, actualProd, epsilon, "Error: " + (expectedProd - actualProd));
      }
   }

    */

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
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC B = new DMatrixSparseCSC(1, 1);
         CommonOps_DSCC.scale(alpha, A, B);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(maxSize, maxSize);

         nativeA.set(A);
         nativeB.scale(alpha, nativeA);
      }

      for (int i = 0; i < iterations; i++)
      {
         int rows = random.nextInt(maxSize) + 1;
         int cols = random.nextInt(maxSize) + 1;
         matrixSizes += (rows + cols) / 2.0;

         double alpha = RandomNumbers.nextDouble(random, 10.0);
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(rows, cols, sparsity, random);
         DMatrixSparseCSC actual = new DMatrixSparseCSC(rows, cols);
         DMatrixSparseCSC expected = new DMatrixSparseCSC(rows, cols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(rows, cols);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(rows, cols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.scale(alpha, nativeA);
         nativeB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DSCC.scale(alpha, A, expected);
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
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(rows, cols, sparsity, random);
         DMatrixSparseCSC actual = new DMatrixSparseCSC(rows, cols);
         DMatrixSparseCSC expected = new DMatrixSparseCSC(rows, cols);

         NativeSparseMatrix nativeB = new NativeSparseMatrix(rows, cols);

         nativeTime -= System.nanoTime();
         nativeB.scale(alpha, A);
         nativeB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DSCC.scale(alpha, A, expected);
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
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC AB = new DMatrixSparseCSC(maxSize, maxSize);
         CommonOps_DSCC.mult(A, B, AB);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeAB = new NativeSparseMatrix(maxSize, maxSize);

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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(aRows, aCols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(aCols, bCols, sparsity, random);
         DMatrixSparseCSC actual = new DMatrixSparseCSC(aRows, bCols);
         DMatrixSparseCSC expected = new DMatrixSparseCSC(aRows, bCols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(aRows, aCols);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(aCols, bCols);
         NativeSparseMatrix nativeAB = new NativeSparseMatrix(aRows, bCols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAB.mult(nativeA, nativeB);
         nativeAB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DSCC.mult(A, B, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         NativeSparseMatrix matrix = new NativeSparseMatrix(1, 1);

         assertDoesNotThrow(() -> matrix.mult(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(3, 7)));
         assertDoesNotThrow(() -> matrix.mult(new NativeSparseMatrix(0, 3), new NativeSparseMatrix(3, 7)));
         assertDoesNotThrow(() -> matrix.mult(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(3, 0)));
         assertDoesNotThrow(() -> matrix.mult(new NativeSparseMatrix(5, 0), new NativeSparseMatrix(0, 7)));
         assertThrows(IllegalArgumentException.class, () -> matrix.mult(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(8, 7)));
         assertThrows(IllegalArgumentException.class, () -> matrix.mult(new NativeSparseMatrix(5, 13), new NativeSparseMatrix(8, 7)));
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

      IGrowArray gx = new IGrowArray();
      DGrowArray gw = new DGrowArray();

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC AB = new DMatrixSparseCSC(maxSize, maxSize);
         DMatrixSparseCSC APlusAB = new DMatrixSparseCSC(maxSize, maxSize);
         CommonOps_DSCC.mult(A, B, AB);
         CommonOps_DSCC.add(1.0, A, 1.0, AB, APlusAB, gx, gw);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeAPlusAB = new NativeSparseMatrix(maxSize, maxSize);

         nativeA.set(A);
         nativeB.set(B);
         nativeAPlusAB.multAdd(nativeA, nativeB);
         nativeAPlusAB.get(AB);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         int bCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols + bCols) / 3.0;

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(aRows, aCols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(aCols, bCols, sparsity, random);
         DMatrixSparseCSC AB = new DMatrixSparseCSC(aRows, bCols);
         DMatrixSparseCSC actual = new DMatrixSparseCSC(aRows, bCols);
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(aRows, bCols, sparsity, random);
         DMatrixSparseCSC original = new DMatrixSparseCSC(expected);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(aRows, aCols);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(aCols, bCols);
         NativeSparseMatrix nativeAB = new NativeSparseMatrix(aRows, bCols);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAB.set(expected);
         nativeAB.multAdd(nativeA, nativeB);
         nativeAB.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DSCC.mult(A, B, AB);
         CommonOps_DSCC.add(1.0, AB, 1.0, original, expected, gx, gw);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(3, 7)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(0, 7).multAdd(new NativeSparseMatrix(0, 3), new NativeSparseMatrix(3, 7)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(5, 0).multAdd(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(3, 0)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(5, 0), new NativeSparseMatrix(0, 7)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(8, 7)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(5, 13), new NativeSparseMatrix(8, 7)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(6, 3), new NativeSparseMatrix(3, 7)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(5, 7).multAdd(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(3, 5)));
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

         DMatrixSparseCSC AtBASparse = new DMatrixSparseCSC(maxSize, maxSize);
         DMatrixSparseCSC sparseA = new DMatrixSparseCSC(maxSize, maxSize);
         DMatrixSparseCSC sparseB = new DMatrixSparseCSC(maxSize, maxSize);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeAtBA = new NativeSparseMatrix(maxSize, maxSize);

         ConvertDMatrixStruct.convert(A, sparseA, epsilon);
         ConvertDMatrixStruct.convert(B, sparseB, epsilon);

         nativeA.set(sparseA);
         nativeB.set(sparseB);
         nativeAtBA.multQuad(nativeA, nativeB);
         nativeAtBA.get(AtBASparse);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols) / 2.0;

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(aRows, aCols, aRows * aCols, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(aRows, aRows, aRows * aRows, random);
         DMatrixSparseCSC actual = new DMatrixSparseCSC(aCols, aCols);
         DMatrixSparseCSC expected = new DMatrixSparseCSC(aCols, aCols);
         DMatrixSparseCSC tempBA = new DMatrixSparseCSC(aRows, aCols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(aRows, aCols);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(aRows, aRows);
         NativeSparseMatrix nativeAtBA = new NativeSparseMatrix(aCols, aCols);

         IGrowArray gw = new IGrowArray();
         DGrowArray gx = new DGrowArray();
         ejmlTime -= System.nanoTime();
         CommonOps_DSCC.mult(B, A, tempBA, gw, gx);
         CommonOps_DSCC.multTransA(A, tempBA, expected, gw, gx);
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
         assertDoesNotThrow(() -> new NativeSparseMatrix(3, 3).multQuad(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(8, 8).multQuad(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(5, 5)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(3, 3).multQuad(new NativeSparseMatrix(0, 3), new NativeSparseMatrix(0, 0)));
         assertDoesNotThrow(() -> new NativeSparseMatrix(0, 0).multQuad(new NativeSparseMatrix(5, 0), new NativeSparseMatrix(5, 5)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeSparseMatrix(3, 3).multQuad(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(6, 5)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(3, 3).multQuad(new NativeSparseMatrix(5, 3), new NativeSparseMatrix(5, 6)));
      }
   }

   @Disabled
   @Test
   public void testInvert()
   {
      Random random = new Random(40L);

      System.out.println("Testing inverting with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> sparseSolver = LinearSolverFactory_DSCC.lu(FillReducing.NONE);
      LinearSolverDense<DMatrixRMaj> denseSolver = LinearSolverFactory_DDRM.lu(0);

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, maxSize * maxSize, -100.0, 100.0, random);
         DMatrixSparseCSC B = new DMatrixSparseCSC(maxSize, maxSize);
         DMatrixSparseCSC identity = CommonOps_DSCC.identity(maxSize);
         DMatrixSparseCSC Bsparse = new DMatrixSparseCSC(maxSize, maxSize);
         sparseSolver.setA(A);
         sparseSolver.solveSparse(identity, B);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(maxSize, maxSize);
         nativeA.set(A);
         nativeB.invert(nativeA);
         nativeB.get(Bsparse);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DMatrixSparseCSC Asparse = RandomMatrices_DSCC.rectangle(aRows, aRows, sparsity, -100.0, 100.0, random);
         DMatrixRMaj A = new DMatrixRMaj(aRows, aRows);
         ConvertDMatrixStruct.convert(Asparse, A);
         DMatrixSparseCSC nativeResult = new DMatrixSparseCSC(aRows, aRows);
         DMatrixRMaj ejmlResult = new DMatrixRMaj(aRows, aRows);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(aRows, aRows);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(aRows, aRows);

         nativeTime -= System.nanoTime();
         nativeA.set(Asparse);
         nativeB.invert(nativeA);
         nativeB.get(nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         denseSolver.setA(A);
         denseSolver.invert(ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(ejmlResult, nativeResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Text exception
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(random.nextInt(maxSize), random.nextInt(maxSize), sparsity, random));
         assertThrows(IllegalArgumentException.class, () -> nativeMatrix.invert(nativeMatrix));
      }
   }


   @Disabled
   @Test
   public void testSolve()
   {
      Random random = new Random(40L);

      System.out.println("Testing solving linear equations with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> solver = LinearSolverFactory_DSCC.lu(FillReducing.NONE);

      for (int i = 0; i < warmumIterations; i++)
      {
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, maxSize * maxSize, random);
         DMatrixSparseCSC x = RandomMatrices_DSCC.rectangle(maxSize, 1, sparsity, random);
         DMatrixSparseCSC b = new DMatrixSparseCSC(maxSize, 1);
         CommonOps_DSCC.mult(A, x, b);
         solver.setA(A);
         solver.solveSparse(b, x);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);
         NativeSparseMatrix nativex = new NativeSparseMatrix(maxSize, 1);
         NativeSparseMatrix nativeb = new NativeSparseMatrix(maxSize, 1);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(aRows, aRows, aRows * aRows, random);
         DMatrixSparseCSC x = RandomMatrices_DSCC.rectangle(aRows, 1, sparsity, random);
         DMatrixSparseCSC b = new DMatrixSparseCSC(aRows, 1);
         CommonOps_DSCC.mult(A, x, b);


         DMatrixSparseCSC nativeResult = new DMatrixSparseCSC(aRows, 1);
         DMatrixSparseCSC ejmlResult = new DMatrixSparseCSC(aRows, 1);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(aRows, aRows);
         NativeSparseMatrix nativex = new NativeSparseMatrix(aRows, 1);
         NativeSparseMatrix nativeb = new NativeSparseMatrix(aRows, 1);

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeb.set(b);
         nativex.solve(nativeA, nativeb);
         nativex.get(nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         solver.setA(A);
         solver.solveSparse(b, ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(x, nativeResult, epsilon);
         MatrixTestTools.assertMatrixEquals(x, ejmlResult, epsilon);
      }

      printTimings(nativeTime, ejmlTime, matrixSizes, iterations);
      System.out.println("--------------------------------------------------------------");

      { // Test exceptions
         assertDoesNotThrow(() -> new NativeSparseMatrix(15, 11).solve(new NativeSparseMatrix(5, 5), new NativeSparseMatrix(5, 1)));
         Class<IllegalArgumentException> expectedType = IllegalArgumentException.class;
         assertThrows(expectedType, () -> new NativeSparseMatrix(15, 11).solve(new NativeSparseMatrix(5, 5), new NativeSparseMatrix(5, 2)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(15, 11).solve(new NativeSparseMatrix(5, 5), new NativeSparseMatrix(4, 1)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(15, 11).solve(new NativeSparseMatrix(5, 4), new NativeSparseMatrix(5, 1)));
         assertThrows(expectedType, () -> new NativeSparseMatrix(15, 11).solve(new NativeSparseMatrix(6, 5), new NativeSparseMatrix(5, 1)));
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

         NativeSparseMatrix randomMatrixA = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(rows, taskSize, sparsity, -50.0, 50.0, random));
         NativeSparseMatrix randomMatrixB = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(taskSize, cols, sparsity, -50.0, 50.0, random));

         NativeSparseMatrix solution = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(fullRows, fullCols, sparsity, -50.0, 50.0, random));

         NativeSparseMatrix expectedSolution = new NativeSparseMatrix(solution);

         NativeSparseMatrix temp = new NativeSparseMatrix(rows, cols);
         temp.mult(randomMatrixA, randomMatrixB);

         expectedSolution.addBlock(temp, rowStart, colStart, 0, 0, rows, cols, 1.0);

         solution.multAddBlock(randomMatrixA, randomMatrixB, rowStart, colStart);

         assertTrue(expectedSolution.isApprox(solution, 1e-6));
      }
   }

   @Test
   public void testAddBlock()
   {
      Random random = new Random(349754);

      { // Test exceptions
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int destRowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int destColStart = RandomNumbers.nextInt(random, 0, fullCols - cols);
         int srcStartRow = 0;
         int srcStartColumn = 0;

         NativeSparseMatrix expectedSolution = new NativeSparseMatrix(fullRows, fullCols);
         NativeSparseMatrix block = new NativeSparseMatrix(rows, cols);

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
   public void testMultTransA()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Arows, Bcols, sparsity, random);
         DMatrixSparseCSC solution = new DMatrixSparseCSC(Acols, Bcols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         IGrowArray gx = new IGrowArray();
         DGrowArray gw = new DGrowArray();

         CommonOps_DSCC.multTransA(A, B, solution, gx, gw);
         nativeSolution.multTransA(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Acols, Bcols);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Arows, Bcols, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(Acols, Bcols, random);

         DMatrixSparseCSC sparseB = new DMatrixSparseCSC(Arows, Bcols);
         DMatrixSparseCSC sparseSolution = new DMatrixSparseCSC(Acols, Bcols);

         ConvertDMatrixStruct.convert(B, sparseB, 1e-5);
         ConvertDMatrixStruct.convert(solution, sparseSolution, epsilon);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(sparseB);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(sparseSolution);

         CommonOps_DSCC.multAddTransA(A, B, solution);
         nativeSolution.multAddTransA(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Acols, Bcols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-5);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Acols, random);
         DMatrixRMaj solution = new DMatrixRMaj(Arows, Brows);

         DMatrixSparseCSC BSparse = new DMatrixSparseCSC(Brows, Acols);
         ConvertDMatrixStruct.convert(B, BSparse, epsilon);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(BSparse);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         CommonOps_DSCC.multTransB(A, B, solution);
         nativeSolution.multTransB(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Arows, Brows);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixRMaj B = RandomMatrices_DDRM.rectangle(Brows, Acols, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(Arows, Brows, random);

         DMatrixSparseCSC BSparse = new DMatrixSparseCSC(Brows, Acols);
         DMatrixSparseCSC solutionSparse = new DMatrixSparseCSC(Arows, Brows);
         ConvertDMatrixStruct.convert(B, BSparse, epsilon);
         ConvertDMatrixStruct.convert(solution, solutionSparse, epsilon);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(BSparse);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(solutionSparse);

         CommonOps_DSCC.multAddTransB(A, B, solution);
         nativeSolution.multAddTransB(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Arows, Brows);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);

         CommonOps_DDRM.extract(B, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insert(nativeB, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset);

         DMatrixSparseCSC nativeADMatrix = new DMatrixSparseCSC(A.getNumRows(), A.getNumCols());
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

         NativeSparseMatrix A = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random));
         NativeSparseMatrix B = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random));

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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);

         CommonOps_DDRM.extract(B, BrowOffset, Brows, BcolOffset, Bcols, A, ArowOffset, AcolOffset);
         nativeA.insert(B, BrowOffset, Brows, BcolOffset, Bcols, ArowOffset, AcolOffset);

         DMatrixSparseCSC nativeADMatrix = new DMatrixSparseCSC(A.getNumRows(), A.getNumCols());
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

         NativeSparseMatrix A = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random));
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random);

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

   @Disabled
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random);
         DMatrixSparseCSC nativeADMatrix = new DMatrixSparseCSC(A);

         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);

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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         NativeSparseMatrix B = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random));

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
   public void testMultScale()
   {
      Random random = new Random(124L);

      for (int i = 0; i < iterations; i++)
      {
         int Arows = RandomNumbers.nextInt(random, 1, 100);
         int Acols = RandomNumbers.nextInt(random, 1, 100);
         int Bcols = RandomNumbers.nextInt(random, 1, 100);

         double scale = RandomNumbers.nextDouble(random, 10000.0);

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Acols, Bcols, sparsity, random);
         DMatrixSparseCSC solution = new DMatrixSparseCSC(Arows, Bcols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         IGrowArray gx = new IGrowArray();
         DGrowArray gw = new DGrowArray();
         CommonOps_DSCC.mult(A, B, solution, gx, gw);
         CommonOps_DSCC.scale(scale, solution, solution);
         nativeSolution.mult(scale, nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Arows, Bcols);
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC solution = new DMatrixSparseCSC(Arows, Acols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         IGrowArray gx = new IGrowArray();
         DGrowArray gw = new DGrowArray();
         CommonOps_DSCC.add(1.0, A, 1.0, B, solution, gx, gw);
         nativeSolution.add(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Arows, Acols);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }

      for (int i = 0; i < iterations; i++)
      { // Test with one of the arguments being the same instance as the owner on which the operation is performed.
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         NativeSparseMatrix A = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(rows, cols, sparsity, random));
         NativeSparseMatrix B = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(rows, cols, sparsity, random));
         NativeSparseMatrix expected = new NativeSparseMatrix(0, 0);
         NativeSparseMatrix actual = new NativeSparseMatrix(0, 0);

         expected.add(A, B);
         DMatrixSparseCSC ejmlExpected = new DMatrixSparseCSC(rows, cols);
         expected.get(ejmlExpected);

         actual.set(B);
         actual.add(A, actual);

         DMatrixSparseCSC ejmlActual = new DMatrixSparseCSC(rows, cols);
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

         NativeSparseMatrix A = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random));
         NativeSparseMatrix B = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random));
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(1, 1);

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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC B = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC solution = new DMatrixSparseCSC(Arows, Acols);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeB = new NativeSparseMatrix(B);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         IGrowArray gx = new IGrowArray();
         DGrowArray gw = new DGrowArray();
         CommonOps_DSCC.add(1.0, A, -1.0, B, solution, gx, gw);
         nativeSolution.subtract(nativeA, nativeB);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Arows, Acols);
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

         NativeSparseMatrix A = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random));
         NativeSparseMatrix B = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(Brows, Bcols, sparsity, random));
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(1, 1);

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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         DMatrixSparseCSC solution = new DMatrixSparseCSC(Acols, Arows);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);
         NativeSparseMatrix nativeSolution = new NativeSparseMatrix(0, 0);

         IGrowArray gx = new IGrowArray();
         CommonOps_DSCC.transpose(A, solution, gx);
         nativeSolution.transpose(nativeA);

         DMatrixSparseCSC nativeSolutionDMatrix = new DMatrixSparseCSC(Acols, Arows);
         nativeSolution.get(nativeSolutionDMatrix);

         MatrixTestTools.assertMatrixEquals(solution, nativeSolutionDMatrix, 1.0e-7);
      }

      { // Text exception
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(random.nextInt(maxSize), random.nextInt(maxSize), sparsity, random));
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);

         NativeSparseMatrix nativeA = new NativeSparseMatrix(A);

         for (int j = 0; j < iterations; j++)
         {
            int row = RandomNumbers.nextInt(random, 0, Arows - 1);
            int col = RandomNumbers.nextInt(random, 0, Acols - 1);

            assertEquals(A.get(row, col), nativeA.get(row, col), 1e-10);
         }
      }

      { // Test exceptions
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(20, 30, sparsity, random));
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

         NativeSparseMatrix nativeA = new NativeSparseMatrix(Arows, Acols);

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
         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(RandomMatrices_DSCC.rectangle(20, 30, sparsity, random));
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
         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         NativeSparseMatrix nativeA = new NativeSparseMatrix(maxSize, maxSize);

         nativeA.set(A);
      }

      double matrixSizes = 0;

      for (int i = 0; i < iterations; i++)
      { // Test set(DMatrixRMaj matrix)
         int Arows = RandomNumbers.nextInt(random, 1, maxSize);
         int Acols = RandomNumbers.nextInt(random, 1, maxSize);

         matrixSizes += (Arows + Acols) / 2.0;

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         NativeSparseMatrix nativeA = new NativeSparseMatrix(Arows, Acols);

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
         DMatrixSparseCSC expected = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);
         DMatrixSparseCSC actual = RandomMatrices_DSCC.rectangle(maxSize, maxSize, sparsity, random);

         NativeSparseMatrix nativeMatrix = new NativeSparseMatrix(random.nextInt(20), random.nextInt(20));

         nativeMatrix.set((Matrix) expected);
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         nativeMatrix.set(RandomMatrices_DSCC.rectangle(random.nextInt(20), random.nextInt(20), sparsity, random));
         nativeMatrix.set((Matrix) new NativeSparseMatrix(expected));
         nativeMatrix.get(actual);
         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);

         nativeMatrix.set(RandomMatrices_DSCC.rectangle(random.nextInt(20), random.nextInt(20), sparsity, random));
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

         DMatrixSparseCSC A = RandomMatrices_DSCC.rectangle(Arows, Acols, sparsity, random);
         NativeSparseMatrix nativeA = new NativeSparseMatrix(Arows, Acols);

         nativeA.set(A);

         DMatrixSparseCSC B = new DMatrixSparseCSC(Arows, Acols);
         MatrixTestTools.assertMatrixEqualsZero(B, 1e-10);

         nativeA.get(B);

         MatrixTestTools.assertMatrixEquals(A, B, 1e-10);
      }
   }

   @Test
   public void testSize()
   {
      NativeSparseMatrix nativeA = new NativeSparseMatrix(0, 0);

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
}
