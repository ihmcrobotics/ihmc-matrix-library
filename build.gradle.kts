plugins {
   id("us.ihmc.ihmc-build")
}

ihmc {
   group = "us.ihmc"
   version = "0.19.3"
   vcsUrl = "https://github.com/ihmcrobotics/ihmc-matrix-library"
   openSource = true

   configureDependencyResolution()
   configurePublications()
}

mainDependencies {
   api("org.ejml:ejml-core:0.39")
   api("org.ejml:ejml-ddense:0.39")

   api("us.ihmc:ihmc-commons:0.35.1")
   api("us.ihmc:euclid:0.22.3")
   api("us.ihmc:ihmc-native-library-loader:2.0.4")
   api("net.sf.trove4j:trove4j:3.0.3")
}

testDependencies {
   api("us.ihmc:euclid-frame:0.22.3")
   api("org.ejml:ejml-simple:0.39")
}
