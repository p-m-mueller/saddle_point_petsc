file(GLOB sources "${CMAKE_SOURCE_DIR}/src/*.c" "${CMAKE_SOURCE_DIR}/incldue/*.h")

add_executable(saddle_point_run ${sources})

target_link_libraries(saddle_point_run PUBLIC
	m
	MPI::MPI_C
	${PETSC_LIBRARIES}
)

target_include_directories(saddle_point_run PUBLIC 
	"${CMAKE_SOURCE_DIR}/include"
	${PETSC_INCLUDE_DIR}
)
