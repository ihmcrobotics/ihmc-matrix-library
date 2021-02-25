%module NativeMatrixLibrary
%javaconst(1);
%include "typemaps.i"
%include "various.i"


%typemap(jtype) int* dims() "java.nio.ByteBuffer"
%typemap(jstype) int* dims() "java.nio.ByteBuffer"
%typemap(jni) int* dims() "jobject"
%typemap(out) int* dims()
%{
    $result = jenv->NewDirectByteBuffer($1, 3 * sizeof(int));
%}
%typemap(javaout) int* dims() {
    return $jnicall;
}


%typemap(jtype) double* "double[]"
%typemap(jstype) double* "double[]"
%typemap(javain) double* "$javainput"
%typemap(jni) double* "jdoubleArray"
%typemap(in) double* {
    $1 = (double*) jenv->GetPrimitiveArrayCritical($input, NULL);
}
%typemap(freearg) double* {
    jenv->ReleasePrimitiveArrayCritical($input, $1, 0);
}

%ignore matrix;

%include "NativeMatrix.h"
%include "NativeNullspaceProjector.h"
%include "NativeKalmanFilter.h"

%{
#include "NativeMatrix.h"
#include "NativeNullspaceProjector.h"
#include "NativeKalmanFilter.h"
%}

