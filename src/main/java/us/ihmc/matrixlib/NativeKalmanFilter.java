package us.ihmc.matrixlib;

import us.ihmc.matrixlib.jni.NativeKalmanFilterImpl;

public class NativeKalmanFilter
{
   public static void predictErrorCovariance(NativeMatrix errorCovariance, NativeMatrix F, NativeMatrix P, NativeMatrix Q)
   {
      if (!NativeKalmanFilterImpl.predictErrorCovariance(errorCovariance.impl, F.impl, P.impl, Q.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public static void computeKalmanGain(NativeMatrix gain, NativeMatrix P, NativeMatrix H, NativeMatrix R)
   {
         if(!NativeKalmanFilterImpl.computeKalmanGain(gain.impl, P.impl, H.impl, R.impl));
         {
            throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
         }
   }
   
   public static void updateState(NativeMatrix nextState, NativeMatrix x, NativeMatrix K , NativeMatrix r)
   {
      if(!NativeKalmanFilterImpl.updateState(nextState.impl, x.impl, K.impl, r.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   public static void updateErrorCovariance(NativeMatrix nextError, NativeMatrix K, NativeMatrix H, NativeMatrix P)
   {
      if(!NativeKalmanFilterImpl.updateErrorCovariance(nextError.impl, K.impl, H.impl, P.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
}
