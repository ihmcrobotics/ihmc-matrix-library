%module NativeMatrixLibrary
%javaconst(1);
%include "typemaps.i"
%include "various.i"


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

%include "NativeMatrix.h"

%{
#include "NativeMatrix.h"
%}

