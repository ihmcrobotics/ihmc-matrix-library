package us.ihmc.matrixlib;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.RandomNumbers;

public class DiagonalMatrixToolsTest
{
   private final double epsilon = 1e-6;

   @Test
   public void testSquareInvert()
   {
      Random random = new Random(1738L);
      int iters = 1000;

      for (int iter = 0; iter < iters; iter++)
      {
         int size = random.nextInt(100);
         DMatrixRMaj matrix = CommonOps_DDRM.identity(size, size);
         DMatrixRMaj invMatrix = new DMatrixRMaj(size, size);
         DMatrixRMaj otherInvMatrix = new DMatrixRMaj(size, size);
         DMatrixRMaj otherInvMatrixB = new DMatrixRMaj(size, size);

         for (int index = 0; index < size; index++)
            matrix.set(index, index, RandomNumbers.nextDouble(random, 10000.0));

         otherInvMatrixB.set(matrix);

         LinearSolverDense<DMatrixRMaj> solver = LinearSolverFactory_DDRM.linear(size);
         solver.setA(matrix);
         solver.invert(invMatrix);

         DiagonalMatrixTools.invertDiagonalMatrix(matrix, otherInvMatrix);
         DiagonalMatrixTools.invertDiagonalMatrix(otherInvMatrixB);

         MatrixTestTools.assertMatrixEquals(invMatrix, otherInvMatrix, epsilon);
         MatrixTestTools.assertMatrixEquals(invMatrix, otherInvMatrixB, epsilon);

         for (int row = 0; row < size; row++)
         {
            for (int col = 0; col < size; col++)
            {
               if (row != col)
                  assertEquals(otherInvMatrix.get(row, col), 0.0, epsilon);
               else
                  assertEquals(otherInvMatrix.get(row, col), 1.0 / matrix.get(row, col), epsilon);
            }
         }

         DiagonalMatrixTools.invertDiagonalMatrix(otherInvMatrix);
         MatrixTestTools.assertMatrixEquals(matrix, otherInvMatrix, epsilon);
      }
   }

   @Test
   public void testPreMult()
   {
      Random random = new Random(1738L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int diagonalRows = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);
         int randomCols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(diagonalRows, interiorCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(diagonalRows, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(interiorCols, randomCols, -5000.0, 5000.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(diagonalRows, randomCols);
         DMatrixRMaj solutionB = new DMatrixRMaj(diagonalRows, randomCols);
         DMatrixRMaj otherSolution = new DMatrixRMaj(diagonalRows, randomCols);

         for (int index = 0; index < Math.min(diagonalRows, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 5000.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, value);
         }

         DiagonalMatrixTools.preMult(diagonal, randomMatrix, solution);
         DiagonalMatrixTools.preMult(diagonalVector, randomMatrix, solutionB);
         CommonOps_DDRM.mult(diagonal, randomMatrix, otherSolution);

         MatrixTestTools.assertMatrixEquals(otherSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(otherSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testPreMultVector()
   {
      Random random = new Random(1738L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int diagonalRows = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);
         int randomCols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(diagonalRows, interiorCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(diagonalRows, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(interiorCols, randomCols, -5000.0, 5000.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(diagonalRows, randomCols);
         DMatrixRMaj otherSolution = new DMatrixRMaj(diagonalRows, randomCols);

         for (int index = 0; index < Math.min(diagonalRows, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 5000.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, value);
         }

         DiagonalMatrixTools.preMult(diagonalVector, randomMatrix, solution);
         CommonOps_DDRM.mult(diagonal, randomMatrix, otherSolution);

         MatrixTestTools.assertMatrixEquals(otherSolution, solution, epsilon);
      }
   }

   @Test
   public void testPreMultAddBlock()
   {
      Random random = new Random(1738L);

      int iters = 100;

      for (int i = 0; i < iters; i++)
      {
         int rows = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);

         int fullRows = RandomNumbers.nextInt(random, rows, 500);
         int fullCols = RandomNumbers.nextInt(random, cols, 500);

         int startRow = RandomNumbers.nextInt(random, 0, fullRows - rows);
         int startCol = RandomNumbers.nextInt(random, 0, fullCols - cols);

         double scalar = RandomNumbers.nextDouble(random, 1000.0);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(rows, interiorCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(rows, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(interiorCols, cols, -100.0, 100.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullRows, fullCols, -10.0, 10.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj solutionC = new DMatrixRMaj(solution);
         DMatrixRMaj solutionD = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         for (int index = 0; index < Math.min(rows, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 100.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(rows, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrix, temp);
         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, rows, cols, 1.0);
         MatrixTools.addMatrixBlock(expectedSolutionB, startRow, startCol, temp, 0, 0, rows, cols, scalar);

         DiagonalMatrixTools.preMultAddBlock(diagonal, randomMatrix, solution, startRow, startCol);
         DiagonalMatrixTools.preMultAddBlock(scalar, diagonal, randomMatrix, solutionB, startRow, startCol);
         DiagonalMatrixTools.preMultAddBlock(diagonalVector, randomMatrix, solutionC, startRow, startCol);
         DiagonalMatrixTools.preMultAddBlock(scalar, diagonalVector, randomMatrix, solutionD, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionD, epsilon);

         MatrixTestTools.assertMatrixEquals(solution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(solutionB, solutionD, epsilon);
      }
   }

   @Test
   public void testPostMult()
   {
      Random random = new Random(1738L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int leadingRows = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);
         int randomCols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(interiorCols, randomCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(interiorCols, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(leadingRows, interiorCols, -5000.0, 5000.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(leadingRows, randomCols);
         DMatrixRMaj solutionB = new DMatrixRMaj(leadingRows, randomCols);
         DMatrixRMaj otherSolution = new DMatrixRMaj(leadingRows, randomCols);

         for (int index = 0; index < Math.min(randomCols, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 5000.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DiagonalMatrixTools.postMult(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.postMult(randomMatrix, diagonalVector, solutionB);
         CommonOps_DDRM.mult(randomMatrix, diagonal, otherSolution);

         MatrixTestTools.assertMatrixEquals(otherSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(otherSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testPostMultVector()
   {
      Random random = new Random(1738L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int leadingRows = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);
         int randomCols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(interiorCols, randomCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(interiorCols, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(leadingRows, interiorCols, -5000.0, 5000.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(leadingRows, randomCols);
         DMatrixRMaj otherSolution = new DMatrixRMaj(leadingRows, randomCols);

         for (int index = 0; index < Math.min(randomCols, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 5000.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DiagonalMatrixTools.postMult(randomMatrix, diagonalVector, solution);
         CommonOps_DDRM.mult(randomMatrix, diagonal, otherSolution);

         MatrixTestTools.assertMatrixEquals(solution, otherSolution, epsilon);
      }
   }

   @Test
   public void testPostMultTransA()
   {
      DMatrixRMaj diagonal = new DMatrixRMaj(2, 4);
      DMatrixRMaj A = new DMatrixRMaj(2, 3);
      DMatrixRMaj solution = new DMatrixRMaj(3, 4);
      DMatrixRMaj expectedSolution = new DMatrixRMaj(3, 4);

      diagonal.set(0, 0, 7.0);
      diagonal.set(1, 1, 8.0);

      A.set(0, 0, 1.0);
      A.set(0, 1, 2.0);
      A.set(0, 2, 3.0);

      A.set(1, 0, 4.0);
      A.set(1, 1, 5.0);
      A.set(1, 2, 6.0);

      DiagonalMatrixTools.postMultTransA(A, diagonal, solution);
      CommonOps_DDRM.multTransA(A, diagonal, expectedSolution);

      MatrixTestTools.assertMatrixEquals(solution, expectedSolution, epsilon);
   }

   @Test
   public void testRandomPostMultTransA()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int leadingRows = RandomNumbers.nextInt(random, 1, 100);
         int interiorCols = RandomNumbers.nextInt(random, 1, 100);
         int randomCols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(interiorCols, randomCols);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(interiorCols, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(interiorCols, leadingRows, -5000.0, 5000.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(leadingRows, randomCols);
         DMatrixRMaj solutionB = new DMatrixRMaj(leadingRows, randomCols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(leadingRows, randomCols);

         for (int index = 0; index < Math.min(randomCols, interiorCols); index++)
         {
            double value = RandomNumbers.nextDouble(random, 5000.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DiagonalMatrixTools.postMultTransA(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.postMultTransA(randomMatrix, diagonalVector, solutionB);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, expectedSolution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testEasyMultInner()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, variables);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DiagonalMatrixTools.postMultTransA(randomMatrix, diagonal, tempJtW);

         // Compute: H += J^T W J
         CommonOps_DDRM.mult(tempJtW, randomMatrix, expectedSolution);

         DiagonalMatrixTools.multInner(randomMatrix, diagonal, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomMultInner()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         double diagonalScalar = RandomNumbers.nextDouble(random, 50.0);
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj solutionB = new DMatrixRMaj(variables, variables);
         DMatrixRMaj solutionC = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(variables, variables);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);

         // Compute: H += J^T W J
         CommonOps_DDRM.mult(tempJtW, randomMatrix, expectedSolution);
         CommonOps_DDRM.multTransA(diagonalScalar, randomMatrix, randomMatrix, expectedSolutionB);

         DiagonalMatrixTools.multInner(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.multInner(randomMatrix, diagonalScalar, solutionB);
         DiagonalMatrixTools.multInner(randomMatrix, diagonalVector, solutionC);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionC, epsilon);
      }
   }

   @Test
   public void testEasyMultOuter()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(variables, taskSize, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, variables);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempWJ = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.mult(randomMatrix, diagonal, tempWJ);
         CommonOps_DDRM.multTransB(tempWJ, randomMatrix, expectedSolution);

         DiagonalMatrixTools.multOuter(randomMatrix, diagonal, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomMultOuter()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);

         double diagonalScalar = RandomNumbers.nextDouble(random, 50.0);
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(variables, taskSize, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj solutionB = new DMatrixRMaj(variables, variables);
         DMatrixRMaj solutionC = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, variables);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(variables, variables);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj tempWJ = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.mult(randomMatrix, diagonal, tempWJ);
         CommonOps_DDRM.multTransB(tempWJ, randomMatrix, expectedSolution);
         CommonOps_DDRM.multTransB(diagonalScalar, randomMatrix, randomMatrix, expectedSolutionB);

         DiagonalMatrixTools.multOuter(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.multOuter(randomMatrix, diagonalScalar, solutionB);
         DiagonalMatrixTools.multOuter(randomMatrix, diagonalVector, solutionC);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionC, epsilon);
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

         double scale = RandomNumbers.nextDouble(random, 100.0);
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, variables, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DiagonalMatrixTools.postMultTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.multAdd(tempJtW, randomMatrix, expectedSolution);
         CommonOps_DDRM.multAdd(scale, tempJtW, randomMatrix, expectedSolutionB);

         DiagonalMatrixTools.multAddInner(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.multAddInner(scale, randomMatrix, diagonal, solutionB);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
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
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, variables, -50, 50, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj solutionD = new DMatrixRMaj(solution);
         DMatrixRMaj solutionE = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.multAdd(tempJtW, randomMatrix, expectedSolution);
         CommonOps_DDRM.multAdd(scale, tempJtW, randomMatrix, expectedSolutionB);

         DiagonalMatrixTools.multAddInner(randomMatrix, diagonal, solution);
         DiagonalMatrixTools.multAddInner(scale, randomMatrix, diagonal, solutionB);
         DiagonalMatrixTools.multAddInner(randomMatrix, diagonalVector, solutionD);
         DiagonalMatrixTools.multAddInner(scale, randomMatrix, diagonalVector, solutionE);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionD, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionE, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionD, epsilon);
         MatrixTestTools.assertMatrixEquals(solutionB, solutionE, epsilon);
      }
   }

   @Test
   public void testRandomMultAddBlockInner()
   {
      Random random = new Random(124L);

      int iters = 100;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int fullVariables = RandomNumbers.nextInt(random, variables, 500);

         int startRow = RandomNumbers.nextInt(random, 0, fullVariables - variables);
         int startCol = RandomNumbers.nextInt(random, 0, fullVariables - variables);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj expectedSolution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50, 50, random);
         DMatrixRMaj solution = new DMatrixRMaj(expectedSolution);
         DMatrixRMaj solutionB = new DMatrixRMaj(expectedSolution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);
         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonalVector, solutionB, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testEasyMultAddBlockInner()
   {
      Random random = new Random(124L);

      int iters = 1000;

      // top left
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 0;
         int startRow = 0;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // top middle
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 1;
         int startRow = 0;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // top right
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 2;
         int startRow = 0;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // middle left
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 0;
         int startRow = 1;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // middle
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 1;
         int startRow = 1;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // middle right
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 2;
         int startRow = 1;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // bottom left
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 0;
         int startRow = 2;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // bottom middle
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 1;
         int startRow = 2;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

      // bottom right
      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;

         int fullVariables = 6;
         int startCol = 2;
         int startRow = 2;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrix = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullVariables, fullVariables, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj tempJtW = new DMatrixRMaj(variables, taskSize);
         DMatrixRMaj temp = new DMatrixRMaj(variables, variables);
         CommonOps_DDRM.multTransA(randomMatrix, diagonal, tempJtW);
         CommonOps_DDRM.mult(tempJtW, randomMatrix, temp);

         MatrixTools.addMatrixBlock(expectedSolution, startRow, startCol, temp, 0, 0, variables, variables, 1.0);

         DiagonalMatrixTools.multAddBlockInner(randomMatrix, diagonal, solution, startRow, startCol);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }

   }

   @Test
   public void testEasyInnerDiagonalMult()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;
         int cols = 5;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(variables, taskSize, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.mult(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMult(randomMatrixA, diagonal, randomMatrixB, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMult()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(variables, taskSize, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj solutionB = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.mult(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMult(randomMatrixA, diagonal, randomMatrixB, solution);
         DiagonalMatrixTools.innerDiagonalMult(randomMatrixA, diagonalVector, randomMatrixB, solutionB);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultVector()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(variables, taskSize, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.mult(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMult(randomMatrixA, diagonalVector, randomMatrixB, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testEasyInnerDiagonalMultTransA()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = 4;
         int taskSize = 3;
         int cols = 5;

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            diagonal.set(index, index, RandomNumbers.nextDouble(random, 50.0));
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multTransA(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMultTransA(randomMatrixA, diagonal, randomMatrixB, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultTransA()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj solutionB = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multTransA(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMultTransA(randomMatrixA, diagonal, randomMatrixB, solution);
         DiagonalMatrixTools.innerDiagonalMultTransA(randomMatrixA, diagonalVector, randomMatrixB, solutionB);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultTransAVector()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = new DMatrixRMaj(variables, cols);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(variables, cols);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multTransA(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMultTransA(randomMatrixA, diagonalVector, randomMatrixB, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultAddTransA()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, cols, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multAddTransA(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMultAddTransA(randomMatrixA, diagonal, randomMatrixB, solution);
         DiagonalMatrixTools.innerDiagonalMultAddTransA(randomMatrixA, diagonalVector, randomMatrixB, solutionB);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionB, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultAddTransAVector()
   {
      Random random = new Random(124L);

      int iters = 1000;

      for (int i = 0; i < iters; i++)
      {
         int variables = RandomNumbers.nextInt(random, 1, 100);
         int taskSize = RandomNumbers.nextInt(random, 1, 100);
         int cols = RandomNumbers.nextInt(random, 1, 100);

         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, variables, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);
         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(variables, cols, -50.0, 50.0, random);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multAddTransA(randomMatrixA, temp, expectedSolution);

         DiagonalMatrixTools.innerDiagonalMultAddTransA(randomMatrixA, diagonalVector, randomMatrixB, solution);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultAddBlockTransA()
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
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, rows, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj solutionC = new DMatrixRMaj(solution);
         DMatrixRMaj solutionD = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         DMatrixRMaj temp2 = new DMatrixRMaj(rows, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multTransA(randomMatrixA, temp, temp2);

         MatrixTools.addMatrixBlock(expectedSolution, rowStart, colStart, temp2, 0, 0, rows, cols, 1.0);
         MatrixTools.addMatrixBlock(expectedSolutionB, rowStart, colStart, temp2, 0, 0, rows, cols, scale);

         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(randomMatrixA, diagonal, randomMatrixB, solution, rowStart, colStart);
         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(randomMatrixA, diagonalVector, randomMatrixB, solutionC, rowStart, colStart);
         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(scale, randomMatrixA, diagonal, randomMatrixB, solutionB, rowStart, colStart);
         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(scale, randomMatrixA, diagonalVector, randomMatrixB, solutionD, rowStart, colStart);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(solution, solutionC, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionD, epsilon);
         MatrixTestTools.assertMatrixEquals(solutionB, solutionD, epsilon);
      }
   }

   @Test
   public void testRandomInnerDiagonalMultAddBlockTransAVector()
   {
      Random random = new Random(124L);

      int iters = 1000;

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
         DMatrixRMaj diagonal = CommonOps_DDRM.identity(taskSize, taskSize);
         DMatrixRMaj diagonalVector = new DMatrixRMaj(taskSize, 1);
         DMatrixRMaj randomMatrixA = RandomMatrices_DDRM.rectangle(taskSize, rows, -50.0, 50.0, random);
         DMatrixRMaj randomMatrixB = RandomMatrices_DDRM.rectangle(taskSize, cols, -50.0, 50.0, random);

         DMatrixRMaj solution = RandomMatrices_DDRM.rectangle(fullRows, fullCols, -50.0, 50.0, random);
         DMatrixRMaj solutionB = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolution = new DMatrixRMaj(solution);
         DMatrixRMaj expectedSolutionB = new DMatrixRMaj(solution);

         for (int index = 0; index < taskSize; index++)
         {
            double value = RandomNumbers.nextDouble(random, 50.0);
            diagonal.set(index, index, value);
            diagonalVector.set(index, 0, value);
         }

         DMatrixRMaj temp = new DMatrixRMaj(taskSize, cols);
         DMatrixRMaj temp2 = new DMatrixRMaj(rows, cols);
         CommonOps_DDRM.mult(diagonal, randomMatrixB, temp);
         CommonOps_DDRM.multTransA(randomMatrixA, temp, temp2);

         MatrixTools.addMatrixBlock(expectedSolution, rowStart, colStart, temp2, 0, 0, rows, cols, 1.0);
         MatrixTools.addMatrixBlock(expectedSolutionB, rowStart, colStart, temp2, 0, 0, rows, cols, scale);

         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(randomMatrixA, diagonalVector, randomMatrixB, solution, rowStart, colStart);
         DiagonalMatrixTools.innerDiagonalMultAddBlockTransA(scale, randomMatrixA, diagonalVector, randomMatrixB, solutionB, rowStart, colStart);

         MatrixTestTools.assertMatrixEquals(expectedSolution, solution, epsilon);
         MatrixTestTools.assertMatrixEquals(expectedSolutionB, solutionB, epsilon);
      }
   }
}
