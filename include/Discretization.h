#ifndef _DISCRETIZATION_H_
#define _DISCRETIZATION_H_

#include <petsc.h>
#include "Utils.h"

typedef struct
{
	PetscInt	dim;
	DM		dm, dmCell;
	Vec		triangles;	// Vector of Triangles
}Mesh;

typedef struct
{
	PetscInt	numCoords;
	PetscReal 	*coords;
	PetscReal	v[2];
	PetscReal	J[4];
	PetscReal	invJ[4];
	PetscReal	detJ;
	PetscInt	numQuadPoints;	// Number of quadrature points
	PetscReal	*quadPoints;	// Quadrature points
	PetscReal	*quadWeight;	// Quadrature weights
}Triangle;

typedef struct
{
	PetscScalar	Ux, Uy;
}Field;

PetscErrorCode MeshCreate(MPI_Comm, const char *, Mesh *);

PetscErrorCode MeshSetupSection(Mesh *);

PetscErrorCode MeshSetupGeometry(Mesh *);

PetscErrorCode MeshDestroy(Mesh *);

PetscErrorCode AssembleOperator_Laplace(Mesh *, Mat *); 
PetscErrorCode AssembleRHS_Laplace(DM, Vec *); 
PetscErrorCode ApplyBC_Laplace(DM, Mat *, Vec *); 

PetscErrorCode AssembleOperator_Constraints(DM, DM, Mat *); 
PetscErrorCode AssembleRHS_Constraints(DM, DM, Vec *);

PetscErrorCode FormStressOperator(PetscScalar *, PetscScalar *, PetscScalar *);

PetscErrorCode FormLaplaceRHS(PetscScalar *, PetscErrorCode(*)(PetscScalar*, PetscScalar*), PetscScalar *); 

PetscErrorCode Phi(const PetscInt, const PetscScalar *, PetscScalar *);
PetscErrorCode GradPhi(const PetscInt, const PetscScalar *, PetscScalar *);

PetscErrorCode ElementStiffnesMatrix(Mesh *, Triangle *, PetscReal *);

#endif
