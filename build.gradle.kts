plugins {
   id("us.ihmc.ihmc-build")
   id("us.ihmc.ihmc-ci") version "7.7"
   id("us.ihmc.ihmc-cd") version "1.24"
}

ihmc {
   group = "us.ihmc"
   version = "0.18.10"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-matrix-library"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")

   api("us.ihmc:ihmc-commons:0.32.0")
   api("us.ihmc:euclid:0.19.1")
   api("us.ihmc:ihmc-native-library-loader:2.0.2")
   api("net.sf.trove4j:trove4j:3.0.3")
}

testDependencies {
   api("us.ihmc:euclid-frame:0.19.1")
   api("org.ejml:ejml-simple:0.39")
}
