package us.ihmc.matrixlib;

import us.ihmc.matrixlib.jni.NativeNullspaceProjectorImpl;

/**
 * {@code NativeNullspaceProjector} can be used to project a first matrix into the nullspace of a
 * second matrix. The entire operation is performed in C++ to maximize performance.
 * 
 * @author Jesper Smith
 */
public class NativeNullspaceProjector
{
   private final NativeNullspaceProjectorImpl impl;

   /**
    * Creates a new instance of a nullspace projector for a given problem size.
    * 
    * @param matrixSize the problem size this calculator can solve.
    */
   public NativeNullspaceProjector(int matrixSize)
   {
      if (matrixSize < 0)
         throw new IllegalArgumentException("Matrix size cannot be negative");
      impl = new NativeNullspaceProjectorImpl(matrixSize);
   }

   /**
    * Projects the matrix {@code a} onto the null-space of {@code b} and stores the result in {@code c}
    * such that</br>
    * b * c == 0</br>
    * This method uses a damped least square approach causing the null-space to grow gradually.
    * <p>
    * The projection is computed as follows:</br>
    * c = a * &Nu;</br>
    * &Nu; = I - b<sup>+</sup>b</br>
    * b<sup>+</sup> = b<sup>T</sup> ( b b<sup>T</sup> - &alpha; I)<sup>-1</sup></br>
    * where &Nu; is the nullspace projector and B<sup>+</sup> the Moore-Penrose pseudo-inverse of b.
    * </p>
    * 
    * @param a     matrix to project. The matrix size is n-by-m where n is unconstrained and m is
    *              constrained to the problem size of this calculator. Not modified.
    * @param b     matrix to compute the null-space of. The matrix size is p-by-m where p is
    *              unconstrained and m is constrained to the problem size of this calculator. Not
    *              modified.
    * @param c     where the result is stored. The matrix is resized to a n-by-m matrix. Modified.
    * @param alpha damping value.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void project(NativeMatrix a, NativeMatrix b, NativeMatrix c, double alpha)
   {
      if (!impl.projectOnNullSpace(a.impl, b.impl, c.impl, alpha))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
}
