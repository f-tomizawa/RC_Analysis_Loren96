MODULE lorenz96
  use common
  PRIVATE

  PUBLIC :: tinteg_rk4

  INTEGER,PARAMETER,PUBLIC :: nx=40         ! number of grid points
  REAL(r_size),SAVE,PUBLIC :: dt=0.005d0    ! time of one time step
  REAL(r_size),SAVE,PUBLIC :: force=8.0d0   ! F term
  REAL(r_size),SAVE,PUBLIC :: oneday=0.2d0  ! time for one day
CONTAINS

SUBROUTINE tinteg_rk4(kt,xin,xout)
  IMPLICIT NONE

  INTEGER,INTENT(IN) :: kt
  REAL(r_size),INTENT(IN)  :: xin(1:nx)
  REAL(r_size),INTENT(OUT) :: xout(1:nx)
  REAL(r_size),ALLOCATABLE :: x(:),xtmp(:),q1(:),q2(:),q3(:),q4(:)
  INTEGER :: k
!--[1.1.1] allocation --------------------------------------------------
  ALLOCATE( x(1:nx) ) ! nx : number of grid points
  ALLOCATE( xtmp(1:nx) )
  ALLOCATE( q1(1:nx) )
  ALLOCATE( q2(1:nx) )
  ALLOCATE( q3(1:nx) )
  ALLOCATE( q4(1:nx) )
!--[1.1.2] time integration --------------------------------------------
  x(:) = xin(:)
!>>>>> TIME INTEGRATION START
  DO k=1,kt
    xtmp(:) = x(:)
    CALL lorenz96_core(xtmp,q1)
    xtmp(:) = x(:) + 0.5d0 * q1(:)
    CALL lorenz96_core(xtmp,q2)
    xtmp(:) = x(:) + 0.5d0 * q2(:)
    CALL lorenz96_core(xtmp,q3)
    xtmp(:) = x(:) + q3(:)
    CALL lorenz96_core(xtmp,q4)
    x(:) = x(:) + ( q1(:) + 2.0d0 * q2(:) + 2.0d0 * q3(:) + q4(:) ) / 6.0d0
  END DO
!<<<<< TIME INTEGRATION END
  xout(:) = x(:)
!--[1.1.3] tidy up -----------------------------------------------------
  DEALLOCATE( xtmp,q1,q2,q3,q4 )

  RETURN
END SUBROUTINE tinteg_rk4

SUBROUTINE lorenz96_core(xin,xout)
  IMPLICIT NONE

  REAL(r_size),INTENT(IN) :: xin(1:nx)
  REAL(r_size),INTENT(OUT) :: xout(1:nx)
  INTEGER :: i

  xout(1) = xin(nx) * ( xin(2) - xin(nx-1) ) - xin(1) + force
  xout(2) = xin(1) * ( xin(3) - xin(nx) ) - xin(2) + force
  DO i=3,nx-1
    xout(i) = xin(i-1) * ( xin(i+1) - xin(i-2) ) - xin(i) + force
  END DO
  xout(nx) = xin(nx-1) * ( xin(1) - xin(nx-2) ) - xin(nx) + force

  xout(:) = dt * xout(:)

  RETURN
END SUBROUTINE lorenz96_core

END MODULE lorenz96
