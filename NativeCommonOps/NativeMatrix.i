%module NativeMatrixLibrary
%javaconst(1);
%include "typemaps.i"
%include "various.i"

/*
%typemap(jtype) double* data() "java.nio.ByteBuffer"
%typemap(jstype) double* data() "java.nio.ByteBuffer"
%typemap(jni) double* data() "jobject"
%typemap(out) double* data()
%{
    $result = jenv->NewDirectByteBuffer($1, arg1->size() * sizeof(double));
%}
%typemap(javaout) double* data() {
    return $jnicall;
}
*/

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


%include "NativeMatrix.h"

%{
#include "NativeMatrix.h"
%}

