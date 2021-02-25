package us.ihmc.matrixlib;

import us.ihmc.matrixlib.jni.NativeKalmanFilterImpl;

public class NativeKalmanFilter
{
   /**
    * Computes {@code F * P * F' + Q} and stores the result in errorCovariance.
    * 
    * @param errorCovariance Result matrix
    * @param F Square matrix
    * @param P Square, upper diagonal matrix
    * @param Q Square diagonal matrix
    */
   public static void predictErrorCovariance(NativeMatrix errorCovariance, NativeMatrix F, NativeMatrix P, NativeMatrix Q)
   {
      if (!NativeKalmanFilterImpl.predictErrorCovariance(errorCovariance.impl, F.impl, P.impl, Q.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * 
    * Computes {@code P * H' * inverse(H * P * H' + R)} and stores the result in gain.
    * 
    * @param gain Result Matrix
    * @param P Square, upper diagonal matrix
    * @param H System jacobian
    * @param R Square diagonal matrix
    */
   public static void computeKalmanGain(NativeMatrix gain, NativeMatrix P, NativeMatrix H, NativeMatrix R)
   {
         if(!NativeKalmanFilterImpl.computeKalmanGain(gain.impl, P.impl, H.impl, R.impl))
         {
            throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
         }
   }
   
   /**
    * Computes {@code x + K * r} and stores the result in nextState
    * 
    * @param nextState Result matrix
    * @param x Row matrix
    * @param K Kalman gain (@see computeKalmanGain(NativeMatrix gain, NativeMatrix P, NativeMatrix H, NativeMatrix R)
    * @param r Row matrix
    */
   public static void updateState(NativeMatrix nextState, NativeMatrix x, NativeMatrix K , NativeMatrix r)
   {
      if(!NativeKalmanFilterImpl.updateState(nextState.impl, x.impl, K.impl, r.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes {@code (identity - K * H) * P} and stores the result in nextError
    * 
    * @param nextError  Result matrix
    * @param K  Kalman gain (@see computeKalmanGain(NativeMatrix gain, NativeMatrix P, NativeMatrix H, NativeMatrix R)
    * @param H System jacobian
    * @param P Square, upper diagonal matrix
    */
   public static void updateErrorCovariance(NativeMatrix nextError, NativeMatrix K, NativeMatrix H, NativeMatrix P)
   {
      if(!NativeKalmanFilterImpl.updateErrorCovariance(nextError.impl, K.impl, H.impl, P.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
}
