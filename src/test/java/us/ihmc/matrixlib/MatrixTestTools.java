package us.ihmc.matrixlib;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.ejml.data.DMatrix;

public class MatrixTestTools
{

   public static void assertMatrixEquals(DMatrix expected, DMatrix actual, double delta)
   {
      MatrixTestTools.assertMatrixEquals("", expected, actual, delta);
   }

   public static void assertMatrixEqualsZero(DMatrix matrix, double epsilon)
   {
      MatrixTestTools.assertMatrixEqualsZero("", matrix, epsilon);
   }

   public static void assertMatrixEqualsZero(String message, DMatrix matrix, double epsilon)
   {
      int numberOfRows = matrix.getNumRows();
      int numberOfColumns = matrix.getNumCols();

      for (int row = 0; row < numberOfRows; row++)
      {
         for (int column = 0; column < numberOfColumns; column++)
         {
            assertEquals(0.0, matrix.get(row, column), epsilon, message);
         }
      }
   }

   public static void assertMatrixEquals(String message, DMatrix expected, DMatrix actual, double delta)
   {
      assertEquals(expected.getNumRows(), actual.getNumRows(), message);
      assertEquals(expected.getNumCols(), actual.getNumCols(), message);

      for (int i = 0; i < expected.getNumRows(); i++)
      {
         for (int j = 0; j < expected.getNumCols(); j++)
         {
            assertEquals(expected.get(i, j), actual.get(i, j), delta, message + " index (" + i + ", " + j + ")");
         }
      }
   }
   
   public static void setDiagonal(DMatrix matrix, int startRow, int startCol, int size, double value)
   {
     
      for(int i = 0; i < size; i++)
      {
         matrix.set(startRow + i, startCol + i, value);
      }
   }
   

}
