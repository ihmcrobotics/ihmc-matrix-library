plugins {
   id("us.ihmc.ihmc-build") version "0.21.0"
   id("us.ihmc.ihmc-ci") version "5.3"
}

ihmc {
   group = "us.ihmc"
   version = "0.17.1"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-matrix-library"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")

   api("us.ihmc:ihmc-commons:0.29.0")
   api("us.ihmc:euclid:0.15.1")
   api("us.ihmc:ihmc-native-library-loader:1.2.1")
}

testDependencies {
   api("us.ihmc:euclid-frame:0.15.1")
}
