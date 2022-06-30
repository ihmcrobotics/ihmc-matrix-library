plugins {
   id("us.ihmc.ihmc-build")
   id("us.ihmc.ihmc-ci") version "7.6"
   id("us.ihmc.ihmc-cd") version "1.23"
}

ihmc {
   group = "us.ihmc"
   version = "0.18.7"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-matrix-library"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")

   api("us.ihmc:ihmc-commons:0.31.0")
   api("us.ihmc:euclid:0.18.1")
   api("us.ihmc:ihmc-native-library-loader:1.3.1")
}

testDependencies {
   api("us.ihmc:euclid-frame:0.18.1")
   api("org.ejml:ejml-simple:0.39")
}
