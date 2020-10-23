package us.ihmc.matrixlib;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.Conversions;

public class NullspaceProjectorTest
{

   private static final int maxSize = 80;
   private static final int iterations = 5000;
   private static final double epsilon = 1.0e-8;
   

   // Make volatile to force operation order
   private volatile long nativeTime = 0;
   private volatile long ejmlTime = 0;
   
   @Test
   public void testProjectOnNullspace()
   {
      Random random = new Random(40L);

      System.out.println("Testing projecting nullspace with random matrices...");

      nativeTime = 0;
      ejmlTime = 0;
      double matrixSizes = 0;

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int dofs = random.nextInt(maxSize) + 1;
         
         matrixSizes += (aRows + dofs + 2 * dofs) / 4.0;

         DMatrixRMaj A = RandomMatrices_DDRM.rectangle(aRows, dofs, random);
         DMatrixRMaj b =  RandomMatrices_DDRM.rectangle(dofs, dofs, random);
         
         NativeMatrix nativeResultMatrix = new NativeMatrix(aRows, dofs);
         DMatrixRMaj nativeResult = new DMatrixRMaj(aRows, dofs);

         DMatrixRMaj ejmlResult = new DMatrixRMaj(aRows, dofs);
         
         NativeMatrix nativeA = new NativeMatrix(A);
         NativeMatrix nativeb = new NativeMatrix(aRows, 1);
         
         NullspaceProjector nullspaceProjector = new NullspaceProjector(dofs);
         
         double alpha = 0.5;
         
         nativeTime -= System.nanoTime();
         nativeA.set(A);
         nativeb.set(b);
         nullspaceProjector.project(nativeA, nativeb, nativeResultMatrix, alpha);
         nativeResultMatrix.get(nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         NativeCommonOps.projectOnNullspace(A, b, ejmlResult, alpha);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(ejmlResult, nativeResult, epsilon);
      }

      System.out.println("NativeMatrix took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("NativeCommonOps took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("NativeMatrix takes " + 100.0 * nativeTime / ejmlTime + "% of NativeCommonOps time.\n");
   }
}
