package us.ihmc.matrixlib;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.RandomNumbers;
import us.ihmc.euclid.referenceFrame.FramePoint3D;
import us.ihmc.euclid.referenceFrame.ReferenceFrame;
import us.ihmc.euclid.referenceFrame.tools.EuclidFrameRandomTools;

public class MatrixToolsTest
{
   @Test
   public void testSetToNaNDenseMatrix()
   {
      DMatrixRMaj test = new DMatrixRMaj(3, 3);
      MatrixTools.setToNaN(test);

      for (int i = 0; i < 3; i++)
      {
         for (int j = 0; j < 3; j++)
         {
            assertTrue(Double.isNaN(test.get(i, j)));
         }
      }
   }

   @Test
   public void testSetToZeroDenseMatrix()
   {
      DMatrixRMaj test = new DMatrixRMaj(3, 3);
      MatrixTools.setToZero(test);

      for (int i = 0; i < 3; i++)
      {
         for (int j = 0; j < 3; j++)
         {
            assertEquals(0.0, test.get(i, j), 1e-34);
         }
      }
   }

   @Test
   public void testSetMatrixColumnFromArrayDenseMatrix()
   {
      DMatrixRMaj test = new DMatrixRMaj(2, 2);

      double[] col = new double[] {3.0, 4.0};

      MatrixTools.setMatrixColumnFromArray(test, 1, col);

      assertEquals(col[0], test.get(0, 1), 1e-8);
      assertEquals(col[1], test.get(1, 1), 1e-8);

   }

   @Test
   public void testDiffDenseMatrixIntIntDenseMatrix()
   {
      double[][] vals = new double[][] {{1.0}, {2.0}, {4.0}, {8.0}, {16.0}, {32.0}};
      DMatrixRMaj test = new DMatrixRMaj(vals);

      DMatrixRMaj res = new DMatrixRMaj(2, 1);

      MatrixTools.diff(test, 2, 3, res);

      assertEquals(4.0, res.get(0, 0), 1e-8);
      assertEquals(8.0, res.get(1, 0), 1e-8);

   }

   @Test
   public void testDiffDoubleArrayDenseMatrix()
   {
      double[] vals = new double[] {1.0, 3.0, 4.0, 9.0, 16.0, 32.0};
      double[] expected = new double[] {2.0, 1.0, 5.0, 7.0, 16.0};
      DMatrixRMaj res = new DMatrixRMaj(5, 1);

      MatrixTools.diff(vals, res);

      for (int i = 0; i < 5; i++)
      {
         assertEquals(expected[i], res.get(i, 0), 1e-8);
      }
   }

   @Test
   public void testDiffDoubleArrayDenseMatrixRange()
   {
      double[] vals = new double[] {Double.NaN, Double.NaN, 1.0, 3.0, 4.0, 9.0, 16.0, 32.0, Double.NaN};
      double[] expected = new double[] {2.0, 1.0, 5.0, 7.0, 16.0};
      DMatrixRMaj res = new DMatrixRMaj(5, 1);

      MatrixTools.diff(vals, 2, 6, res);

      for (int i = 0; i < 5; i++)
      {
         assertEquals(expected[i], res.get(i, 0), 1e-8);
      }
   }

   @Test
   public void testRemoveRow()
   {
      Random random = new Random(3216516L);
      for (int i = 0; i < 20; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         int indexOfRowToRemove = RandomNumbers.nextInt(random, 0, randomMatrix.getNumRows() - 1);
         DMatrixRMaj expectedMatrix = new DMatrixRMaj(numRows - 1, numCols);

         for (int rowIndex = 0; rowIndex < numRows - 1; rowIndex++)
         {
            for (int colIndex = 0; colIndex < numCols; colIndex++)
            {
               if (rowIndex >= indexOfRowToRemove)
                  expectedMatrix.set(rowIndex, colIndex, randomMatrix.get(rowIndex + 1, colIndex));
               else
                  expectedMatrix.set(rowIndex, colIndex, randomMatrix.get(rowIndex, colIndex));
            }
         }

         DMatrixRMaj matrixToTest = new DMatrixRMaj(randomMatrix);
         MatrixTools.removeRow(matrixToTest, indexOfRowToRemove);

         boolean areMatricesEqual = MatrixFeatures_DDRM.isEquals(expectedMatrix, matrixToTest, 1.0e-10);
         assertTrue(areMatricesEqual);
      }
   }

   @Test
   public void testSetRow()
   {
      Random random = new Random(1738L);
      for (int i = 0; i < 20; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         DMatrixRMaj randomRow = RandomMatrices_DDRM.rectangle(1, numCols, 1.0, 100.0, random);
         int indexOfRowToSet = RandomNumbers.nextInt(random, 0, numRows - 1);

         DMatrixRMaj expectedMatrix = new DMatrixRMaj(randomMatrix);
         DMatrixRMaj matrixToTest = new DMatrixRMaj(randomMatrix);

         for (int j = 0; j < numCols; j++)
            expectedMatrix.set(indexOfRowToSet, j, randomRow.get(0, j));

         MatrixTools.setRow(randomRow, indexOfRowToSet, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(1, numCols, 1.0, 100.0, random);
         indexOfRowToSet = RandomNumbers.nextInt(random, 0, numRows - 1);
         double randomMultiplier = RandomNumbers.nextDouble(random, 1, 100);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.set(indexOfRowToSet, j, randomMultiplier * randomRow.get(0, j));

         MatrixTools.setRow(randomMultiplier, randomRow, indexOfRowToSet, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         int numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);
         indexOfRowToSet = RandomNumbers.nextInt(random, 0, numRows - 1);
         int indexOfOriginRow = RandomNumbers.nextInt(random, 0, numOriginRows - 1);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.set(indexOfRowToSet, j, randomRow.get(indexOfOriginRow, j));

         MatrixTools.setRow(indexOfOriginRow, randomRow, indexOfRowToSet, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);
         indexOfRowToSet = RandomNumbers.nextInt(random, 0, numRows - 1);
         indexOfOriginRow = RandomNumbers.nextInt(random, 0, numOriginRows - 1);

         randomMultiplier = RandomNumbers.nextDouble(random, 1, 100);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.set(indexOfRowToSet, j, randomMultiplier * randomRow.get(indexOfOriginRow, j));

         MatrixTools.setRow(indexOfOriginRow, randomMultiplier, randomRow, indexOfRowToSet, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);

         int numOfRowsToSet = RandomNumbers.nextInt(random, 1, Math.min(numOriginRows, numRows));
         int[] originRowIndices = RandomNumbers.nextIntArray(random, numOfRowsToSet, 1, numOriginRows - 1);
         int[] destRowIndices = RandomNumbers.nextIntArray(random, numOfRowsToSet, 1, numRows - 1);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
         {
            for (int k = 0; k < numOfRowsToSet; k++)
               expectedMatrix.set(destRowIndices[k], j, randomRow.get(originRowIndices[k], j));
         }

         MatrixTools.setRows(originRowIndices, randomRow, destRowIndices, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);
      }
   }

   @Test
   public void testAddRow()
   {
      Random random = new Random(1738L);
      for (int i = 0; i < 20; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         DMatrixRMaj randomRow = RandomMatrices_DDRM.rectangle(1, numCols, 1.0, 100.0, random);
         int indexOfRowToAdd = RandomNumbers.nextInt(random, 0, numRows - 1);

         DMatrixRMaj expectedMatrix = new DMatrixRMaj(randomMatrix);
         DMatrixRMaj matrixToTest = new DMatrixRMaj(randomMatrix);

         for (int j = 0; j < numCols; j++)
            expectedMatrix.add(indexOfRowToAdd, j, randomRow.get(0, j));

         MatrixTools.addRow(randomRow, indexOfRowToAdd, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(1, numCols, 1.0, 100.0, random);
         indexOfRowToAdd = RandomNumbers.nextInt(random, 0, numRows - 1);
         double randomMultiplier = RandomNumbers.nextDouble(random, 1, 100);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.add(indexOfRowToAdd, j, randomMultiplier * randomRow.get(0, j));

         MatrixTools.addRow(randomMultiplier, randomRow, indexOfRowToAdd, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         int numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);
         indexOfRowToAdd = RandomNumbers.nextInt(random, 0, numRows - 1);
         int indexOfOriginRow = RandomNumbers.nextInt(random, 0, numOriginRows - 1);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.add(indexOfRowToAdd, j, randomRow.get(indexOfOriginRow, j));

         MatrixTools.addRow(indexOfOriginRow, randomRow, indexOfRowToAdd, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);
         indexOfRowToAdd = RandomNumbers.nextInt(random, 0, numRows - 1);
         indexOfOriginRow = RandomNumbers.nextInt(random, 0, numOriginRows - 1);

         randomMultiplier = RandomNumbers.nextDouble(random, 1, 100);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
            expectedMatrix.add(indexOfRowToAdd, j, randomMultiplier * randomRow.get(indexOfOriginRow, j));

         MatrixTools.addRow(indexOfOriginRow, randomMultiplier, randomRow, indexOfRowToAdd, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);

         numRows = RandomNumbers.nextInt(random, 1, 100);
         numOriginRows = RandomNumbers.nextInt(random, 1, 100);
         numCols = RandomNumbers.nextInt(random, 1, 100);
         randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         randomRow = RandomMatrices_DDRM.rectangle(numOriginRows, numCols, 1.0, 100.0, random);

         int numOfRowsToSet = RandomNumbers.nextInt(random, 1, Math.min(numOriginRows, numRows));
         int[] originRowIndices = RandomNumbers.nextIntArray(random, numOfRowsToSet, 1, numOriginRows - 1);
         int[] destRowIndices = RandomNumbers.nextIntArray(random, numOfRowsToSet, 1, numRows - 1);

         expectedMatrix = new DMatrixRMaj(randomMatrix);
         matrixToTest = new DMatrixRMaj(randomMatrix);
         for (int j = 0; j < numCols; j++)
         {
            for (int k = 0; k < numOfRowsToSet; k++)
               expectedMatrix.add(destRowIndices[k], j, randomRow.get(originRowIndices[k], j));
         }

         MatrixTools.addRows(originRowIndices, randomRow, destRowIndices, matrixToTest);
         MatrixTestTools.assertMatrixEquals(expectedMatrix, matrixToTest, 1.0e-10);
      }
   }

   @Test
   public void testRemoveColumn()
   {
      Random random = new Random(3216516L);
      for (int i = 0; i < 20; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         int indexOfColumnToRemove = RandomNumbers.nextInt(random, 0, randomMatrix.getNumCols() - 1);
         DMatrixRMaj expectedMatrix = new DMatrixRMaj(numRows, numCols - 1);

         for (int colIndex = 0; colIndex < numCols - 1; colIndex++)
         {
            for (int rowIndex = 0; rowIndex < numRows; rowIndex++)
            {
               if (colIndex >= indexOfColumnToRemove)
                  expectedMatrix.set(rowIndex, colIndex, randomMatrix.get(rowIndex, colIndex + 1));
               else
                  expectedMatrix.set(rowIndex, colIndex, randomMatrix.get(rowIndex, colIndex));
            }
         }

         DMatrixRMaj matrixToTest = new DMatrixRMaj(randomMatrix);
         MatrixTools.removeColumn(matrixToTest, indexOfColumnToRemove);

         for (int colIndex = 0; colIndex < numCols - 1; colIndex++)
         {
            DMatrixRMaj expectedMatrixColumn = new DMatrixRMaj(numRows, 1);
            DMatrixRMaj randomMatrixColumn = new DMatrixRMaj(numRows, 1);

            int originalColumnIndex = colIndex;
            if (colIndex >= indexOfColumnToRemove)
               originalColumnIndex++;

            CommonOps_DDRM.extractColumn(expectedMatrix, colIndex, expectedMatrixColumn);
            CommonOps_DDRM.extractColumn(randomMatrix, originalColumnIndex, randomMatrixColumn);
            boolean areMatricesEqual = MatrixFeatures_DDRM.isEquals(randomMatrixColumn, expectedMatrixColumn, 1.0e-10);
            assertTrue(areMatricesEqual);
         }

         boolean areMatricesEqual = MatrixFeatures_DDRM.isEquals(expectedMatrix, matrixToTest, 1.0e-10);
         assertTrue(areMatricesEqual);
      }
   }

   @Test
   public void testRemoveZeroRows()
   {
      Random random = new Random(3216516L);
      for (int i = 0; i < 200; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         int randomNumberOfZeroRows = RandomNumbers.nextInt(random, 0, 5);
         int[] indicesOfZeroRows = RandomNumbers.nextIntArray(random, randomNumberOfZeroRows, 0, randomMatrix.getNumRows() - 1);

         // Switching to a set to remove duplicates
         HashSet<Integer> filterForDuplicate = new HashSet<>();
         for (int zeroRowIndex : indicesOfZeroRows)
            filterForDuplicate.add(zeroRowIndex);

         indicesOfZeroRows = new int[filterForDuplicate.size()];
         int counter = 0;
         for (int filteredZeroRow : filterForDuplicate)
            indicesOfZeroRows[counter++] = filteredZeroRow;

         Arrays.sort(indicesOfZeroRows);

         for (int zeroRowIndex : indicesOfZeroRows)
         {
            for (int columnIndex = 0; columnIndex < numCols; columnIndex++)
            {
               randomMatrix.set(zeroRowIndex, columnIndex, 0.0);
            }
         }
         DMatrixRMaj expectedMatrix = new DMatrixRMaj(randomMatrix);
         for (int j = indicesOfZeroRows.length - 1; j >= 0; j--)
            MatrixTools.removeRow(expectedMatrix, indicesOfZeroRows[j]);

         DMatrixRMaj matrixToTest = new DMatrixRMaj(randomMatrix);
         MatrixTools.removeZeroRows(matrixToTest, 1.0e-12);

         boolean areMatricesEqual = MatrixFeatures_DDRM.isEquals(expectedMatrix, matrixToTest, 1.0e-10);
         assertTrue(areMatricesEqual);
      }
   }

   @Test
   public void testScaleTranspose() throws Exception
   {
      Random random = new Random(165156L);
      for (int i = 0; i < 200; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 1, 100);
         int numCols = RandomNumbers.nextInt(random, 1, 100);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(numRows, numCols, 1.0, 100.0, random);
         double randomAlpha = RandomNumbers.nextDouble(random, 100.0);
         DMatrixRMaj expectedMatrix = new DMatrixRMaj(numCols, numRows);
         DMatrixRMaj actualMatrix = new DMatrixRMaj(numCols, numRows);

         CommonOps_DDRM.transpose(randomMatrix, expectedMatrix);
         CommonOps_DDRM.scale(randomAlpha, expectedMatrix);

         MatrixTools.scaleTranspose(randomAlpha, randomMatrix, actualMatrix);

         boolean areMatricesEqual = MatrixFeatures_DDRM.isEquals(expectedMatrix, actualMatrix, 1.0e-10);
         assertTrue(areMatricesEqual);
      }
   }

   @Test
   public void testInsertFrameTupleIntoEJMLVector()
   {
      Random random = new Random(3216516L);
      for (int i = 0; i < 1000; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 3, 100);
         DMatrixRMaj matrixToTest = RandomMatrices_DDRM.rectangle(numRows, 1, 1.0, 100.0, random);
         FramePoint3D framePointToInsert = EuclidFrameRandomTools.nextFramePoint3D(random, ReferenceFrame.getWorldFrame(), 100.0, 100.0, 100.0);
         int startRowToInsertFrameTuple = RandomNumbers.nextInt(random, 0, numRows - 3);
         framePointToInsert.get(startRowToInsertFrameTuple, matrixToTest);

         assertEquals(framePointToInsert.getX(), matrixToTest.get(startRowToInsertFrameTuple + 0, 0), 1.0e-10);
         assertEquals(framePointToInsert.getY(), matrixToTest.get(startRowToInsertFrameTuple + 1, 0), 1.0e-10);
         assertEquals(framePointToInsert.getZ(), matrixToTest.get(startRowToInsertFrameTuple + 2, 0), 1.0e-10);
      }
   }

   @Test
   public void testExtractFrameTupleFromEJMLVector()
   {
      Random random = new Random(3216516L);
      for (int i = 0; i < 1000; i++)
      {
         int numRows = RandomNumbers.nextInt(random, 3, 100);
         DMatrixRMaj matrixToExtractFrom = RandomMatrices_DDRM.rectangle(numRows, 1, 1.0, 100.0, random);
         FramePoint3D framePointToTest = new FramePoint3D(null, -1.0, -1.0, -1.0);
         int startRowToExtractFrameTuple = RandomNumbers.nextInt(random, 0, numRows - 3);
         framePointToTest.setIncludingFrame(ReferenceFrame.getWorldFrame(), startRowToExtractFrameTuple, matrixToExtractFrom);

         assertEquals(framePointToTest.getReferenceFrame(), ReferenceFrame.getWorldFrame());
         assertEquals(framePointToTest.getX(), matrixToExtractFrom.get(startRowToExtractFrameTuple + 0, 0), 1.0e-10);
         assertEquals(framePointToTest.getY(), matrixToExtractFrom.get(startRowToExtractFrameTuple + 1, 0), 1.0e-10);
         assertEquals(framePointToTest.getZ(), matrixToExtractFrom.get(startRowToExtractFrameTuple + 2, 0), 1.0e-10);
      }
   }

   @Test
   public void testCheckDenseMatrixDimensions()
   {
      Random ran = new Random(124L);

      int nTests = 500;
      int maxRows = 1000;
      int maxColumns = 1000;
      for (int i = 0; i < nTests; i++)
      {
         int rows = ran.nextInt(maxRows);
         int columns = ran.nextInt(maxColumns);

         // these should not throw exceptions
         try
         {
            DMatrixRMaj testm = new DMatrixRMaj(rows, columns);
            MatrixTools.checkMatrixDimensions(testm, rows, columns);
         }
         catch (Throwable e)
         {
            fail();
         }
      }

   }

   public void testMultAddBlockTransA()
   {
      Random random = new Random(124L);

      int iters = 100;

      for (int i = 0; i < iters; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         double scale = RandomNumbers.nextDouble(random, 1000.0);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, rows, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         DMatrixRMaj temp = new DMatrixRMaj(rows, cols);
         CommonOps_DDRM.multTransA(randomMatrixA, randomMatrixB, temp);
         MatrixTools.addMatrixBlock(expectedSolution, rowStart, colStart, temp, 0, 0, rows, cols, 1.0);
         MatrixTools.addMatrixBlock(expectedSolutionB, rowStart, colStart, temp, 0, 0, rows, cols, scale);

         MatrixTools.multAddBlockTransA(randomMatrixA, randomMatrixB, solution, rowStart, colStart);
         MatrixTools.multAddBlockTransA(scale, randomMatrixA, randomMatrixB, solutionB, rowStart, colStart);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, 1e-6);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, 1e-6);
      }
   }

   public void testMultAddBlock()
   {
      Random random = new Random(124L);

      int iters = 100;

      for (int i = 0; i < iters; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         int rowStart = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int colStart = RandomNumbers.nextInt(random, 0, fullCols - cols);

         double scale = RandomNumbers.nextDouble(random, 1000.0);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(rows, taskSize, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         DMatrixRMaj temp = new DMatrixRMaj(rows, cols);
         CommonOps_DDRM.mult(randomMatrixA, randomMatrixB, temp);
         MatrixTools.addMatrixBlock(expectedSolution, rowStart, colStart, temp, 0, 0, rows, cols, 1.0);
         MatrixTools.addMatrixBlock(expectedSolutionB, rowStart, colStart, temp, 0, 0, rows, cols, scale);

         MatrixTools.multAddBlock(randomMatrixA, randomMatrixB, solution, rowStart, colStart);
         MatrixTools.multAddBlock(scale, randomMatrixA, randomMatrixB, solutionB, rowStart, colStart);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, 1e-6);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, 1e-6);
      }
   }

   @Test
   public void testRandomMultAddBlockInnerWithScalar()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int fullVariables = RandomNumbers.nextInt(random, variables, 500);

         int startRow = RandomNumbers.nextInt(random, 0, fullVariables - variables);
         int startCol = RandomNumbers.nextInt(random, 0, fullVariables - variables);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         double diagonalValue = RandomNumbers.nextDouble(random, 50.0);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj expectedSolution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50, 50, random);
         DMatrixRMaj solution = new DMatrixRMaj(expectedSolution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, diagonalValue);
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         MatrixTools.multAddBlockInner(diagonalValue, randomMatrix, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, 1e-6);
      }
   }

   @Test
   public void testEasyMultAddInner()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         double diagonalScalar = RandomNumbers.nextDouble(random, 100.0);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, variables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.transpose(randomMatrix, tempJtW);
         CommonOps_DDRM.multAdd(diagonalScalar, tempJtW, randomMatrix, expectedSolution);

         MatrixTools.multAddInner(diagonalScalar, randomMatrix, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, 1e-6);
      }
   }

   @Test
   public void testRandomMultAddInner()
   {
      Random random = new Random(124L);

      int iters = 100;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         double scale = RandomNumbers.nextDouble(random, 100.0);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, variables, -50, 50, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.transpose(randomMatrix, tempJtW);
         CommonOps_DDRM.multAdd(scale, tempJtW, randomMatrix, expectedSolution);

         MatrixTools.multAddInner(scale, randomMatrix, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, 1e-6);
      }
   }

   @Test
   public void testSwapRows()
   {
      Random random = new Random(46357);

      for (int iteration = 0; iteration < 1000; iteration++)
      {
         int numRow = random.nextInt(50) + 1;
         int numCol = random.nextInt(50) + 1;
         DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(numRow, numCol, random);
         DMatrixRMaj expected = new DMatrixRMaj(actual);

         int i = random.nextInt(numRow);
         int j = random.nextInt(numRow);

         CommonOps_DDRM.extract(actual, i, i + 1, 0, numCol, expected, j, 0);
         CommonOps_DDRM.extract(actual, j, j + 1, 0, numCol, expected, i, 0);

         MatrixTools.swapRows(i, j, actual);

         MatrixTestTools.assertMatrixEquals(expected, actual, 0.0);
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapRows(-1, j, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapRows(actual.getNumRows(), j, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapRows(i, -1, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapRows(i, actual.getNumRows(), actual));
      }
   }

   @Test
   public void testSwapColumns()
   {
      Random random = new Random(46357);

      for (int iteration = 0; iteration < 1000; iteration++)
      {
         int numRow = random.nextInt(50) + 1;
         int numCol = random.nextInt(50) + 1;
         DMatrixRMaj actual = RandomMatrices_DDRM.rectangle(numRow, numCol, random);
         DMatrixRMaj expected = new DMatrixRMaj(actual);

         int i = random.nextInt(numCol);
         int j = random.nextInt(numCol);

         CommonOps_DDRM.extract(actual, 0, numRow, i, i + 1, expected, 0, j);
         CommonOps_DDRM.extract(actual, 0, numRow, j, j + 1, expected, 0, i);

         MatrixTools.swapColumns(i, j, actual);

         MatrixTestTools.assertMatrixEquals(expected, actual, 0.0);
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapColumns(-1, j, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapColumns(actual.getNumCols(), j, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapColumns(i, -1, actual));
         assertThrows(IllegalArgumentException.class, () -> MatrixTools.swapColumns(i, actual.getNumCols(), actual));
      }
   }
}
