package us.ihmc.matrixlib;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Test;

public class MatrixTestToolsTest
{

   @Test
   public void testAssertDMatrixRMajEquals()
   {
      final int ITERATIONS = 1000;
      final double EPSILON = 0.000001;
      final int MATRIX_SIZE_BOUND = 100;

      Random random1 = new Random(4275L);
      Random random2 = new Random(4275L);

      for (int i = 0; i < ITERATIONS; i++)
      {

         int n1 = random1.nextInt(MATRIX_SIZE_BOUND);
         int m1 = random1.nextInt(MATRIX_SIZE_BOUND);
         DMatrixRMaj matrix1 = new DMatrixRMaj(n1, m1, true, randomDoubleArray(random1, n1 * m1));

         int n2 = random2.nextInt(MATRIX_SIZE_BOUND);
         int m2 = random2.nextInt(MATRIX_SIZE_BOUND);
         DMatrixRMaj matrix2 = new DMatrixRMaj(n2, m2, true, randomDoubleArray(random2, n2 * m2));

         MatrixTestTools.assertMatrixEquals(matrix1, matrix2, EPSILON);
         MatrixTestTools.assertMatrixEquals("testAssertDMatrixRMajEquals", matrix1, matrix2, EPSILON);
      }
   }

   @Test
   public void testAssertMatrixEqualsZero()
   {
      final int ITERATIONS = 1000;
      final double EPSILON = 0.000001;
      final int MATRIX_SIZE_BOUND = 100;

      Random random = new Random(4276L);

      for (int i = 0; i < ITERATIONS; i++)
      {
         DMatrixRMaj matrix = new DMatrixRMaj(random.nextInt(MATRIX_SIZE_BOUND), random.nextInt(MATRIX_SIZE_BOUND));

         MatrixTestTools.assertMatrixEqualsZero(matrix, EPSILON);
         MatrixTestTools.assertMatrixEqualsZero("testAssertMatrixEqualsZero", matrix, EPSILON);
      }
   }

   private double[] randomDoubleArray(Random random, int length)
   {
      final double LARGE_VALUE = 4294967296.0; //Cannot use MIN_VALUE here because of overflows
      double[] array = new double[length];
      for (int i = 0; i < length; i++)
      {
         array[i] = -LARGE_VALUE + 2 * LARGE_VALUE * random.nextDouble();
      }
      return array;
   }
}
