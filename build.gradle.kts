plugins {
   id("us.ihmc.ihmc-build")
   id("us.ihmc.ihmc-ci") version "8.3"
   id("us.ihmc.ihmc-cd") version "1.26"
}

ihmc {
   group = "us.ihmc"
   version = "0.19.0"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-matrix-library"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")

   api("us.ihmc:ihmc-commons:0.34.0")
   api("us.ihmc:euclid:0.22.2")
   api("us.ihmc:ihmc-native-library-loader:2.0.3")
   api("net.sf.trove4j:trove4j:3.0.3")
}

testDependencies {
   api("us.ihmc:euclid-frame:0.22.2")
   api("org.ejml:ejml-simple:0.39")
}
