static const char help[] = "Saddle point example.\n\n";

#include <petsc.h>

typedef struct
{
	PetscScalar uu, tt;
}UserContext;

extern PetscErrorCode ComputeJacobian(KSP, Mat, Mat, void*);
extern PetscErrorCode ComputeRHS(KSP, Vec, void*);

int main(int argc, char **argv)
{

	KSP		ksp;
	DM		da;
	UserContext 	user;
	PetscErrorCode 	ierr;

	ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;	

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);


	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return ierr;
}

PetscErrorCode ComputeJacobian(KSP ksp, Mat J, Mat jac, void *ctx)
{

	return 0;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{

	return 0;
}
