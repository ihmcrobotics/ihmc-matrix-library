package us.ihmc.matrixlib;

import static org.junit.jupiter.api.Assertions.fail;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

public class NativeKalmanFilterTest
{

   private static final int ITERATIONS = 50;
   private static final double EPSILON = 1.0E-10;
   private static final Random random = new Random(86526826L);


   
   
   @Test
   public void testPredictErrorCovariance()
   {
      for (int i = 0; i < ITERATIONS; i++)
      {
         int n = random.nextInt(100) + 1;

         DMatrixRMaj Fdense = RandomMatrices_DDRM.rectangle(n, n, -1.0, 1.0, random);
         NativeMatrix F = new NativeMatrix(Fdense);
         DMatrixRMaj Pdense = RandomMatrices_DDRM.symmetric(n, 0.1, 1.0, random);
         NativeMatrix P = new NativeMatrix(Pdense);
         DMatrixRMaj Qdense = RandomMatrices_DDRM.diagonal(n, 0.1, 1.0, random);
         NativeMatrix Q = new NativeMatrix(Qdense);

         NativeMatrix actual = new NativeMatrix(n, n);
         
         NativeKalmanFilter.predictErrorCovariance(actual, F, P, Q);

         SimpleMatrix Psimple = new SimpleMatrix(Pdense);
         SimpleMatrix Fsimple = new SimpleMatrix(Fdense);
         SimpleMatrix Qsimple = new SimpleMatrix(Qdense);
         DMatrixRMaj expected = Fsimple.mult(Psimple.mult(Fsimple.transpose())).plus(Qsimple).getMatrix();

         MatrixTestTools.assertMatrixEquals(expected, actual, EPSILON);
      }
   }

   @Test
   public void testUpdateErrorCovariance()
   {
      for (int i = 0; i < ITERATIONS; i++)
      {
         int n = random.nextInt(100) + 1;
         int m = random.nextInt(100) + 1;

         DMatrixRMaj K = RandomMatrices_DDRM.rectangle(m, n, -1.0, 1.0, random);
         DMatrixRMaj H = RandomMatrices_DDRM.rectangle(n, m, -1.0, 1.0, random);
         DMatrixRMaj P = RandomMatrices_DDRM.symmetric(m, 0.1, 1.0, random);
         
         
         NativeMatrix Knative = new NativeMatrix(K);
         NativeMatrix Hnative = new NativeMatrix(H);
         NativeMatrix Pnative = new NativeMatrix(P);
         
         NativeMatrix actual = new NativeMatrix(m ,m);
         NativeKalmanFilter.updateErrorCovariance(actual, Knative, Hnative, Pnative);

         SimpleMatrix Psimple = new SimpleMatrix(P);
         SimpleMatrix Hsimple = new SimpleMatrix(H);
         SimpleMatrix Ksimple = new SimpleMatrix(K);
         SimpleMatrix IKH = SimpleMatrix.identity(m).minus(Ksimple.mult(Hsimple));
         DMatrixRMaj expected = IKH.mult(Psimple).getMatrix();

         MatrixTestTools.assertMatrixEquals(expected, actual, EPSILON);
      }
   }

   @Test
   public void testComputeKalmanGain()
   {
      for (int i = 0; i < ITERATIONS; i++)
      {
         int n = random.nextInt(100) + 1;
         int m = random.nextInt(100) + 1;

         DMatrixRMaj P = RandomMatrices_DDRM.symmetric(m, 0.1, 1.0, random);
         DMatrixRMaj H = RandomMatrices_DDRM.rectangle(n, m, -1.0, 1.0, random);
         DMatrixRMaj R = RandomMatrices_DDRM.diagonal(n, 1.0, 100.0, random);
         
         DMatrixRMaj Rdiag = new DMatrixRMaj(n, 1);
         CommonOps_DDRM.extractDiag(R, Rdiag);
         
         SimpleMatrix Psimple = new SimpleMatrix(P);
         SimpleMatrix Hsimple = new SimpleMatrix(H);
         SimpleMatrix Rsimple = new SimpleMatrix(R);
         SimpleMatrix toInvert = Hsimple.mult(Psimple.mult(Hsimple.transpose())).plus(Rsimple);
         if (Math.abs(toInvert.determinant()) < 1.0e-5)
         {
            fail("Poorly conditioned matrix. Change random seed or skip. Determinant is " + toInvert.determinant());
         }
         SimpleMatrix inverse = toInvert.invert();
         DMatrixRMaj expected = Psimple.mult(Hsimple.transpose()).mult(inverse).getMatrix();
         
         
         NativeMatrix actual = new NativeMatrix(m, n);         
         NativeKalmanFilter.computeKalmanGain(actual, new NativeMatrix(P), new NativeMatrix(H), new NativeMatrix(Rdiag));



         MatrixTestTools.assertMatrixEquals(expected, actual, EPSILON);
      }
   }

   @Test
   public void testUpdateState()
   {
      for (int i = 0; i < ITERATIONS; i++)
      {
         int n = random.nextInt(100) + 1;
         int m = random.nextInt(100) + 1;

         DMatrixRMaj x = RandomMatrices_DDRM.rectangle(n, 1, -1.0, 1.0, random);
         DMatrixRMaj K = RandomMatrices_DDRM.rectangle(n, m, -1.0, 1.0, random);
         DMatrixRMaj r = RandomMatrices_DDRM.rectangle(m, 1, -1.0, 1.0, random);

         NativeMatrix actual = new NativeMatrix(n, 1);
         NativeKalmanFilter.updateState(actual, new NativeMatrix(x), new NativeMatrix(K), new NativeMatrix(r));

         SimpleMatrix rSimple = new SimpleMatrix(r);
         SimpleMatrix Ksimple = new SimpleMatrix(K);
         SimpleMatrix xSimple = new SimpleMatrix(x);
         DMatrixRMaj expected = xSimple.plus(Ksimple.mult(rSimple)).getMatrix();

         MatrixTestTools.assertMatrixEquals(expected, actual, EPSILON);
      }
   }
   
}
