package us.ihmc.matrixlib;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.Conversions;

public class NativeMatrixTest
{
   private static final int maxSize = 80;
   private static final int warmumIterations = 2000;
   private static final int iterations = 5000;
   private static final double epsilon = 1.0e-8;
   

   // Make volatile to force operation order
   private volatile long nativeTime = 0;
   private volatile long ejmlTime = 0;

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

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
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

         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeB.set(B);
         nativeAtBA.multQuad(nativeA, nativeB);
         nativeAtBA.get(actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps_DDRM.mult(B, A, tempBA);
         CommonOps_DDRM.multTransA(A, tempBA, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
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

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
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

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
   }

   public static void main(String[] args)
   {
      int size = 500;
      Random random = new Random(40L);
      DMatrixRMaj A = RandomMatrices_DDRM.rectangle(size, size, random);
      DMatrixRMaj B = RandomMatrices_DDRM.rectangle(size, size, random);
      DMatrixRMaj AtBA = new DMatrixRMaj(size, size);

      System.out.println("Running...");

      while (true)
      {
         NativeCommonOps.multQuad(A, B, AtBA);
      }
   }
}
