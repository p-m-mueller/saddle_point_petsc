#ifndef _MESHFUNCTIONS_H_
#define _MESHFUNCTIONS_H_

#include <petsc.h>


#define GAUSS_POINTS 4
#define DIM 2
#define NODES_PER_ELEMENT 4
#define U_DOF 2

typedef struct
{
	PetscInt i, j;
}Index2d;

typedef struct 
{
	PetscScalar 	gp_coords[DIM*GAUSS_POINTS];
	PetscScalar	gp_weights[GAUSS_POINTS];
}ElementProperties;

typedef struct
{
	PetscScalar	Ux, Uy;
}Field;

PetscErrorCode SetupDMDA(PetscInt, PetscInt, DM *);

PetscErrorCode GetElementCoords(DMDACoor2d **, PetscInt, PetscInt, PetscScalar []);

PetscErrorCode ConstructGaussQuadratureQ12D(PetscInt *, PetscScalar [][DIM], PetscScalar []);

PetscErrorCode ConstructQ12D_Ni(PetscScalar [DIM], PetscScalar [NODES_PER_ELEMENT]); 

PetscErrorCode ConstructQ12D_GNi(PetscScalar [DIM], PetscScalar [][NODES_PER_ELEMENT]); 

PetscErrorCode ConstructQ12D_GNx(PetscScalar [][NODES_PER_ELEMENT], PetscScalar *, PetscScalar [][NODES_PER_ELEMENT], PetscScalar *);

PetscErrorCode AssembleOperator_Laplace(DM, Mat *); 
PetscErrorCode AssembleRHS_Laplace(DM, Vec *); 
PetscErrorCode ApplyBC_Laplace(DM, Mat *, Vec *); 

PetscErrorCode AssembleOperator_Constraints(DM, DM, Mat *); 
PetscErrorCode AssembleRHS_Constraints(DM, DM, Vec *);

PetscErrorCode FormStressOperatorQ12D(PetscScalar *, PetscScalar *, PetscScalar *);

PetscErrorCode FormLaplaceRHSQ12D(PetscScalar *, PetscErrorCode(*)(PetscScalar*, PetscScalar*), PetscScalar *); 

PetscErrorCode DMDAGetElementEqnums(PetscInt, PetscInt, MatStencil [NODES_PER_ELEMENT*U_DOF]);

PetscErrorCode FormRHS(PetscScalar *, PetscScalar *);

#endif
