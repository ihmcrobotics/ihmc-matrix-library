/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */


#ifndef SWIGJAVA
#define SWIGJAVA
#endif



#ifdef __cplusplus
/* SwigValueWrapper is described in swig.swg */
template<typename T> class SwigValueWrapper {
  struct SwigMovePointer {
    T *ptr;
    SwigMovePointer(T *p) : ptr(p) { }
    ~SwigMovePointer() { delete ptr; }
    SwigMovePointer& operator=(SwigMovePointer& rhs) { T* oldptr = ptr; ptr = 0; delete oldptr; ptr = rhs.ptr; rhs.ptr = 0; return *this; }
  } pointer;
  SwigValueWrapper& operator=(const SwigValueWrapper<T>& rhs);
  SwigValueWrapper(const SwigValueWrapper<T>& rhs);
public:
  SwigValueWrapper() : pointer(0) { }
  SwigValueWrapper& operator=(const T& t) { SwigMovePointer tmp(new T(t)); pointer = tmp; return *this; }
  operator T&() const { return *pointer.ptr; }
  T *operator&() { return pointer.ptr; }
};

template <typename T> T SwigValueInit() {
  return T();
}
#endif

/* -----------------------------------------------------------------------------
 *  This section contains generic SWIG labels for method/variable
 *  declarations/attributes, and other compiler dependent labels.
 * ----------------------------------------------------------------------------- */

/* template workaround for compilers that cannot correctly implement the C++ standard */
#ifndef SWIGTEMPLATEDISAMBIGUATOR
# if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x560)
#  define SWIGTEMPLATEDISAMBIGUATOR template
# elif defined(__HP_aCC)
/* Needed even with `aCC -AA' when `aCC -V' reports HP ANSI C++ B3910B A.03.55 */
/* If we find a maximum version that requires this, the test would be __HP_aCC <= 35500 for A.03.55 */
#  define SWIGTEMPLATEDISAMBIGUATOR template
# else
#  define SWIGTEMPLATEDISAMBIGUATOR
# endif
#endif

/* inline attribute */
#ifndef SWIGINLINE
# if defined(__cplusplus) || (defined(__GNUC__) && !defined(__STRICT_ANSI__))
#   define SWIGINLINE inline
# else
#   define SWIGINLINE
# endif
#endif

/* attribute recognised by some compilers to avoid 'unused' warnings */
#ifndef SWIGUNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define SWIGUNUSED __attribute__ ((__unused__))
#   else
#     define SWIGUNUSED
#   endif
# elif defined(__ICC)
#   define SWIGUNUSED __attribute__ ((__unused__))
# else
#   define SWIGUNUSED
# endif
#endif

#ifndef SWIG_MSC_UNSUPPRESS_4505
# if defined(_MSC_VER)
#   pragma warning(disable : 4505) /* unreferenced local function has been removed */
# endif
#endif

#ifndef SWIGUNUSEDPARM
# ifdef __cplusplus
#   define SWIGUNUSEDPARM(p)
# else
#   define SWIGUNUSEDPARM(p) p SWIGUNUSED
# endif
#endif

/* internal SWIG method */
#ifndef SWIGINTERN
# define SWIGINTERN static SWIGUNUSED
#endif

/* internal inline SWIG method */
#ifndef SWIGINTERNINLINE
# define SWIGINTERNINLINE SWIGINTERN SWIGINLINE
#endif

/* exporting methods */
#if defined(__GNUC__)
#  if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#    ifndef GCC_HASCLASSVISIBILITY
#      define GCC_HASCLASSVISIBILITY
#    endif
#  endif
#endif

#ifndef SWIGEXPORT
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   if defined(STATIC_LINKED)
#     define SWIGEXPORT
#   else
#     define SWIGEXPORT __declspec(dllexport)
#   endif
# else
#   if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#     define SWIGEXPORT __attribute__ ((visibility("default")))
#   else
#     define SWIGEXPORT
#   endif
# endif
#endif

/* calling conventions for Windows */
#ifndef SWIGSTDCALL
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   define SWIGSTDCALL __stdcall
# else
#   define SWIGSTDCALL
# endif
#endif

/* Deal with Microsoft's attempt at deprecating C standard runtime functions */
#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif

/* Deal with Microsoft's attempt at deprecating methods in the standard C++ library */
#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif

/* Deal with Apple's deprecated 'AssertMacros.h' from Carbon-framework */
#if defined(__APPLE__) && !defined(__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES)
# define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif

/* Intel's compiler complains if a variable which was never initialised is
 * cast to void, which is a common idiom which we use to indicate that we
 * are aware a variable isn't used.  So we just silence that warning.
 * See: https://github.com/swig/swig/issues/192 for more discussion.
 */
#ifdef __INTEL_COMPILER
# pragma warning disable 592
#endif


/* Fix for jlong on some versions of gcc on Windows */
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
  typedef long long __int64;
#endif

/* Fix for jlong on 64-bit x86 Solaris */
#if defined(__x86_64)
# ifdef _LP64
#   undef _LP64
# endif
#endif

#include <jni.h>
#include <stdlib.h>
#include <string.h>


/* Support for throwing Java exceptions */
typedef enum {
  SWIG_JavaOutOfMemoryError = 1, 
  SWIG_JavaIOException, 
  SWIG_JavaRuntimeException, 
  SWIG_JavaIndexOutOfBoundsException,
  SWIG_JavaArithmeticException,
  SWIG_JavaIllegalArgumentException,
  SWIG_JavaNullPointerException,
  SWIG_JavaDirectorPureVirtual,
  SWIG_JavaUnknownError
} SWIG_JavaExceptionCodes;

typedef struct {
  SWIG_JavaExceptionCodes code;
  const char *java_exception;
} SWIG_JavaExceptions_t;


static void SWIGUNUSED SWIG_JavaThrowException(JNIEnv *jenv, SWIG_JavaExceptionCodes code, const char *msg) {
  jclass excep;
  static const SWIG_JavaExceptions_t java_exceptions[] = {
    { SWIG_JavaOutOfMemoryError, "java/lang/OutOfMemoryError" },
    { SWIG_JavaIOException, "java/io/IOException" },
    { SWIG_JavaRuntimeException, "java/lang/RuntimeException" },
    { SWIG_JavaIndexOutOfBoundsException, "java/lang/IndexOutOfBoundsException" },
    { SWIG_JavaArithmeticException, "java/lang/ArithmeticException" },
    { SWIG_JavaIllegalArgumentException, "java/lang/IllegalArgumentException" },
    { SWIG_JavaNullPointerException, "java/lang/NullPointerException" },
    { SWIG_JavaDirectorPureVirtual, "java/lang/RuntimeException" },
    { SWIG_JavaUnknownError,  "java/lang/UnknownError" },
    { (SWIG_JavaExceptionCodes)0,  "java/lang/UnknownError" }
  };
  const SWIG_JavaExceptions_t *except_ptr = java_exceptions;

  while (except_ptr->code != code && except_ptr->code)
    except_ptr++;

  jenv->ExceptionClear();
  excep = jenv->FindClass(except_ptr->java_exception);
  if (excep)
    jenv->ThrowNew(excep, msg);
}


/* Contract support */

#define SWIG_contract_assert(nullreturn, expr, msg) if (!(expr)) {SWIG_JavaThrowException(jenv, SWIG_JavaIllegalArgumentException, msg); return nullreturn; } else


#include "NativeMatrix.h"


#ifdef __cplusplus
extern "C" {
#endif

SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1nan_1set(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdouble jarg2) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double arg2 ;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (double)jarg2; 
  if (arg1) (arg1)->nan = arg2;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1nan_1get(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (double) ((arg1)->nan);
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jlong JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_new_1NativeMatrixImpl(JNIEnv *jenv, jclass jcls, jint jarg1, jint jarg2) {
  jlong jresult = 0 ;
  int arg1 ;
  int arg2 ;
  NativeMatrixImpl *result = 0 ;
  
  (void)jenv;
  (void)jcls;
  arg1 = (int)jarg1; 
  arg2 = (int)jarg2; 
  result = (NativeMatrixImpl *)new NativeMatrixImpl(arg1,arg2);
  *(NativeMatrixImpl **)&jresult = result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1resize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jint jarg2, jint jarg3) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int arg2 ;
  int arg3 ;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (int)jarg2; 
  arg3 = (int)jarg3; 
  (arg1)->resize(arg2,arg3);
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1set_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  result = (bool)(arg1)->set(arg2);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1add(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->add(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1subtract(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->subtract(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1mult_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->mult(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1mult_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdouble jarg2, jlong jarg3, jobject jarg3_, jlong jarg4, jobject jarg4_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double arg2 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg4 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg3_;
  (void)jarg4_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (double)jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  arg4 = *(NativeMatrixImpl **)&jarg4; 
  result = (bool)(arg1)->mult(arg2,arg3,arg4);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multAdd(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multAdd(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multTransA(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multTransA(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multAddTransA(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multAddTransA(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multTransB(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multTransB(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multAddTransB(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multAddTransB(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1addBlock(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jint jarg3, jint jarg4, jint jarg5, jint jarg6, jint jarg7, jint jarg8, jdouble jarg9) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  int arg3 ;
  int arg4 ;
  int arg5 ;
  int arg6 ;
  int arg7 ;
  int arg8 ;
  double arg9 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = (int)jarg3; 
  arg4 = (int)jarg4; 
  arg5 = (int)jarg5; 
  arg6 = (int)jarg6; 
  arg7 = (int)jarg7; 
  arg8 = (int)jarg8; 
  arg9 = (double)jarg9; 
  result = (bool)(arg1)->addBlock(arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multAddBlock(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_, jint jarg4, jint jarg5) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  int arg4 ;
  int arg5 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  arg4 = (int)jarg4; 
  arg5 = (int)jarg5; 
  result = (bool)(arg1)->multAddBlock(arg2,arg3,arg4,arg5);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1multQuad(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->multQuad(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1invert(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  result = (bool)(arg1)->invert(arg2);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1solve(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->solve(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1solveCheck(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->solveCheck(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1insert(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jint jarg3, jint jarg4, jint jarg5, jint jarg6, jint jarg7, jint jarg8) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  int arg3 ;
  int arg4 ;
  int arg5 ;
  int arg6 ;
  int arg7 ;
  int arg8 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = (int)jarg3; 
  arg4 = (int)jarg4; 
  arg5 = (int)jarg5; 
  arg6 = (int)jarg6; 
  arg7 = (int)jarg7; 
  arg8 = (int)jarg8; 
  result = (bool)(arg1)->insert(arg2,arg3,arg4,arg5,arg6,arg7,arg8);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1transpose(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  result = (bool)(arg1)->transpose(arg2);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1removeRow(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jint jarg2) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int arg2 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (int)jarg2; 
  result = (bool)(arg1)->removeRow(arg2);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1removeColumn(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jint jarg2) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int arg2 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (int)jarg2; 
  result = (bool)(arg1)->removeColumn(arg2);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1zero(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  (arg1)->zero();
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1containsNaN(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (bool)(arg1)->containsNaN();
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1scale_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdouble jarg2, jlong jarg3, jobject jarg3_) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double arg2 ;
  NativeMatrixImpl *arg3 = (NativeMatrixImpl *) 0 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg3_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (double)jarg2; 
  arg3 = *(NativeMatrixImpl **)&jarg3; 
  result = (bool)(arg1)->scale(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1isAprrox(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jobject jarg2_, jdouble jarg3) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  NativeMatrixImpl *arg2 = (NativeMatrixImpl *) 0 ;
  double arg3 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  (void)jarg2_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = *(NativeMatrixImpl **)&jarg2; 
  arg3 = (double)jarg3; 
  result = (bool)(arg1)->isAprrox(arg2,arg3);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1set_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdoubleArray jarg2, jint jarg3, jint jarg4) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double *arg2 = (double *) 0 ;
  int arg3 ;
  int arg4 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  {
    arg2 = (double*) jenv->GetPrimitiveArrayCritical(jarg2, NULL);
  }
  arg3 = (int)jarg3; 
  arg4 = (int)jarg4; 
  result = (bool)(arg1)->set(arg2,arg3,arg4);
  jresult = (jboolean)result; 
  {
    jenv->ReleasePrimitiveArrayCritical(jarg2, arg2, 0);
  }
  return jresult;
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1get_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdoubleArray jarg2, jint jarg3, jint jarg4) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double *arg2 = (double *) 0 ;
  int arg3 ;
  int arg4 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  {
    arg2 = (double*) jenv->GetPrimitiveArrayCritical(jarg2, NULL);
  }
  arg3 = (int)jarg3; 
  arg4 = (int)jarg4; 
  result = (bool)(arg1)->get(arg2,arg3,arg4);
  jresult = (jboolean)result; 
  {
    jenv->ReleasePrimitiveArrayCritical(jarg2, arg2, 0);
  }
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1min(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (double)(arg1)->min();
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1max(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (double)(arg1)->max();
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1sum(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (double)(arg1)->sum();
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1prod(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (double)(arg1)->prod();
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1scale_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jdouble jarg2) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  double arg2 ;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (double)jarg2; 
  (arg1)->scale(arg2);
}


SWIGEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1set_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jint jarg2, jint jarg3, jdouble jarg4) {
  jboolean jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int arg2 ;
  int arg3 ;
  double arg4 ;
  bool result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (int)jarg2; 
  arg3 = (int)jarg3; 
  arg4 = (double)jarg4; 
  result = (bool)(arg1)->set(arg2,arg3,arg4);
  jresult = (jboolean)result; 
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1get_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jint jarg2, jint jarg3) {
  jdouble jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int arg2 ;
  int arg3 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  arg2 = (int)jarg2; 
  arg3 = (int)jarg3; 
  result = (double)(arg1)->get(arg2,arg3);
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1rows(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jint jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (int)(arg1)->rows();
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1cols(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jint jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (int)(arg1)->cols();
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1size(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  jint jresult = 0 ;
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  result = (int)(arg1)->size();
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_NativeMatrixImpl_1print(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  
  (void)jenv;
  (void)jcls;
  (void)jarg1_;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  (arg1)->print();
}


SWIGEXPORT void JNICALL Java_us_ihmc_matrixlib_jni_NativeMatrixLibraryJNI_delete_1NativeMatrixImpl(JNIEnv *jenv, jclass jcls, jlong jarg1) {
  NativeMatrixImpl *arg1 = (NativeMatrixImpl *) 0 ;
  
  (void)jenv;
  (void)jcls;
  arg1 = *(NativeMatrixImpl **)&jarg1; 
  delete arg1;
}


#ifdef __cplusplus
}
#endif

