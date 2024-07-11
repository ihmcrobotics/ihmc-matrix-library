package us.ihmc.matrixlib;

import us.ihmc.tools.nativelibraries.NativeLibraryDescription;
import us.ihmc.tools.nativelibraries.NativeLibraryLoader;
import us.ihmc.tools.nativelibraries.NativeLibraryWithDependencies;

public class MatrixLibNativeLibrary implements NativeLibraryDescription
{
   @Override
   public String getPackage(OperatingSystem os, Architecture arch)
   {
      String archPackage = "";
      if (arch == Architecture.x64)
      {
         archPackage = switch (os)
         {
            case WIN64 -> "windows-x86_64";
            case LINUX64 -> "linux-x86_64";
            case MACOSX64 -> "macos-x86_64";
         };
      }
      else if (arch == Architecture.arm64)
      {
         archPackage = switch (os)
         {
            case WIN64 -> "windows-arm64"; // Currently unsupported
            case LINUX64 -> "linux-arm64";
            case MACOSX64 -> "macos-arm64";
         };
      }

      return "ihmc-matrix-library.native." + archPackage;
   }

   @Override
   public NativeLibraryWithDependencies getLibraryWithDependencies(OperatingSystem os, Architecture arch)
   {
      switch (os)
      {
         case LINUX64 ->
         {
            return NativeLibraryWithDependencies.fromFilename("libNativeCommonOps.so");
         }
         case WIN64 ->
         {
            return NativeLibraryWithDependencies.fromFilename("NativeCommonOps.dll");
         }
         case MACOSX64 ->
         {
            return NativeLibraryWithDependencies.fromFilename("libNativeCommonOps.jnilib");
         }
      }

      System.out.println("Unsupported platform: " + os.name() + "-" + arch.name());

      return null;   }

   private static boolean loaded = false;

   public static boolean load()
   {
      if (!loaded)
      {
         MatrixLibNativeLibrary lib = new MatrixLibNativeLibrary();
         loaded = NativeLibraryLoader.loadLibrary(lib);
      }
      return loaded;
   }
}
