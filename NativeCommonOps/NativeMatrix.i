%module NativeMatrixLibrary
%javaconst(1);
%include "typemaps.i"
%include "various.i"

%apply unsigned char *NIOBUFFER { double* };


%include "NativeMatrix.h"

%{
#include "NativeMatrix.h"
%}

