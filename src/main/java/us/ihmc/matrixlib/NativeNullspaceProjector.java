package us.ihmc.matrixlib;

import us.ihmc.matrixlib.jni.NativeNullspaceProjectorImpl;

public class NativeNullspaceProjector
{
   private final NativeNullspaceProjectorImpl impl;
   
   public NativeNullspaceProjector(int degreesOfFreedom)
   {
      impl = new NativeNullspaceProjectorImpl(degreesOfFreedom);
   }
   

   /**
    * Projects the matrix {@code a} onto the null-space of {@code b} and stores the result in {@code c}
    * such that</br>
    * b * c == 0</br>
    * This method uses a damped least square approach causing the null-space to grow gradually.
    * 
    * @param a     matrix to project
    * @param b     matrix to compute the null-space of
    * @param c     where the result is stored (modified)
    * @param alpha damping value
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void project(NativeMatrix a, NativeMatrix b, NativeMatrix c, double alpha)
   {

      if(!impl.projectOnNullSpace(a.impl, b.impl, c.impl, alpha))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
}
