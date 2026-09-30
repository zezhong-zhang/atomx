! Matched-potential SBOUND latency and wave probe. The caller supplies the
! potential table, output grid, initial energy and series tolerance.
program probe
  use constants
  implicit none
  real(8) :: rad(ndim), p(ndim), q(ndim), r0(ndim), rv0(ndim)
  real(8) :: energy, guess, eps, t0, t1
  integer :: ngp, ilast, ier, n, k, i, nv, repetitions, repeat
  common /radwf/ rad, p, q, ngp, ilast, ier
  read(*,*) nv, ngp, guess, n, k, eps, repetitions
  if (ngp>ndim.or.nv>ndim.or.repetitions<1) stop 1
  do i=1,nv
    read(*,*) r0(i), rv0(i)
  end do
  do i=1,ngp
    read(*,*) rad(i)
  end do
  call vint(r0, rv0, nv)
  call cpu_time(t0)
  do repeat=1,repetitions
    energy=guess
    call sbound(energy, eps, n, k)
    if (ier/=0) exit
  end do
  call cpu_time(t1)
  write(*,'(a,2i10,2es25.16)') '# ',ier,ilast,energy,(t1-t0)/repetitions
  if (ier/=0) stop 2
  do i=1,ngp
    write(*,'(3es25.16)') rad(i),p(i),q(i)
  end do
end program
