#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Genetic::genetic_algorithm" for configuration ""
set_property(TARGET Genetic::genetic_algorithm APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Genetic::genetic_algorithm PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libgenetic_algorithm.a"
  )

list(APPEND _cmake_import_check_targets Genetic::genetic_algorithm )
list(APPEND _cmake_import_check_files_for_Genetic::genetic_algorithm "${_IMPORT_PREFIX}/lib/libgenetic_algorithm.a" )

# Import target "Genetic::simple-ga-test" for configuration ""
set_property(TARGET Genetic::simple-ga-test APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Genetic::simple-ga-test PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/simple_ga_test"
  )

list(APPEND _cmake_import_check_targets Genetic::simple-ga-test )
list(APPEND _cmake_import_check_files_for_Genetic::simple-ga-test "${_IMPORT_PREFIX}/bin/simple_ga_test" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
