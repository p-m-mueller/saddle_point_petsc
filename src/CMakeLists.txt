file(GLOB sources *.c)

add_executable(saddle_point_run ${sources})

target_link_libraries(saddle_point_run PUBLIC
	MPI::MPI_C
	${PETSC_LIBRARIES}
)

target_include_directories(saddle_point_run PUBLIC 
	${PETSC_INCLUDE_DIR}
)
