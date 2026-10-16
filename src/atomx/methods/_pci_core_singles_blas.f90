! Exact component-tagged native specialization of pinned pCI GPL-3.0 terma4.
! The complete direct plus recoupled exchange rho matrix is retained. Shared
! core/virtual potentials and disjoint core-single outputs permit OpenMP.
subroutine terma4_blas()
  use native_sd_bridge
  implicit none
  integer j,kb,kapb,lb,jb,indb,b0,inn,kn,kapn,ln,jn,n0,nnn,k,kmin,kmax,n,i
  real(c_double),allocatable :: phi(:,:,:),potential(:,:),w(:,:),rho(:,:),v(:),u(:)
  allocate(phi(nhf,nx,nk),potential(nhf,nxx))
  phi=gg+ff
  !$omp parallel num_threads(CASE_THREADS) default(shared) &
  !$omp private(j,kb,kapb,lb,jb,indb,b0,inn,kn,kapn,ln,jn,n0,nnn,k,kmin,kmax,n,i,w,rho,v,u)
  allocate(w(nhf,nxx),rho(nxx,nxx),v(nhf),u(nhf))
  do j=1,ncore
    kb=ko(j)
    call klj(kb,kapb,lb,jb,indb,b0)
    do inn=1,nk
      call indk1(inn,kn,kapn,ln,jn,n0)
      call st(kn,nnn)
      kmin=abs(kapn-kapb); kmax=kapn+kapb-1
      do k=kmin,kmax
        if(mod(ln+lb+k,2)/=0) cycle
        !$omp do schedule(static)
        do n=nnn,nmaxx(inn)
          v=(go(:,j)+fo(:,j))*phi(:,n,inn)
          call density_action(v,u,k)
          potential(:,n)=u
        enddo
        !$omp end do
        !$omp do schedule(dynamic,1)
        do i=1,ncore
          call terma4_blas_core(i,j,inn,k,phi,potential,w,rho,v)
        enddo
        !$omp end do
      enddo
    enddo
  enddo
  deallocate(w,rho,v,u)
  !$omp end parallel
  deallocate(phi,potential)
end subroutine

recursive subroutine terma4_blas_core(i,j,inn,k,phi,potential,w,rho,v)
  use native_sd_bridge
  implicit none
  integer,intent(in) :: i,j,inn,k
  real(c_double),intent(in) :: phi(nhf,nx,nk),potential(nhf,nxx)
  real(c_double),intent(inout) :: w(nhf,nxx),rho(nxx,nxx),v(nhf)
  integer ka,kb,kapa,kapb,la,lb,ja,jb,im,indb,a0,b0
  integer kn,kapn,ln,jn,n0,nnn,km,kapm,lm,jm,m0,nnm,ir,kr,kapr,lr,jr,r0,nnr
  integer k2,k2min,k2max,index2,n,nr,m,cn,cr,cm
  real(c_double) zz,z1,coef
  real(c_double),external :: s,d6j
  ka=ko(i); kb=ko(j)
  call klj(ka,kapa,la,ja,im,a0)
  call klj(kb,kapb,lb,jb,indb,b0)
  call indk1(im,km,kapm,lm,jm,m0)
  call indk1(inn,kn,kapn,ln,jn,n0)
  call st(km,nnm); call st(kn,nnn)
  cm=nmaxx(im)-nnm+1; cn=nmaxx(inn)-nnn+1
  if(cm==0.or.cn==0) return
  do ir=1,nk
    call indk1(ir,kr,kapr,lr,jr,r0)
    call st(kr,nnr)
    if(k<abs(kapm-kapr).or.k>kapm+kapr-1.or.mod(lr+lm+k,2)/=0) cycle
    cr=nmaxx(ir)-nnr+1
    if(cr==0) cycle
    if(i>=j) then
      index2=ipxc(i,j,ir,inn,k)
      if(index2==0) error stop 'complete terma4 direct source absent'
      do n=nnn,nmaxx(inn)
        do nr=nnr,nmaxx(ir)
          rho(nr,n)=xci(nr,n,index2)
        enddo
      enddo
    else
      index2=ipxc(j,i,inn,ir,k)
      if(index2==0) error stop 'complete terma4 direct source absent'
      do n=nnn,nmaxx(inn)
        do nr=nnr,nmaxx(ir)
          rho(nr,n)=xci(n,nr,index2)
        enddo
      enddo
    endif
    zz=(-1d0)**(kapa+kapr+kapb+kapn)/((ja+1d0)*(2*k+1d0))
    k2min=max(abs(kapr-kapb),abs(kapn-kapa))
    k2max=min(kapr+kapb-1,kapn+kapa-1)
    do k2=k2min,k2max
      if(mod(lr+lb,2)/=mod(ln+la,2)) cycle
      z1=(2*k+1d0)*d6j(jr,ja,2*k,jn,jb,2*k2)
      if(i>=j) then
        index2=ipxc(i,j,inn,ir,k2)
      else
        index2=ipxc(j,i,ir,inn,k2)
      endif
      if(index2==0) error stop 'complete terma4 exchange source absent'
      do n=nnn,nmaxx(inn)
        do nr=nnr,nmaxx(ir)
          if(i>=j) then
            rho(nr,n)=rho(nr,n)+z1*xci(n,nr,index2)
          else
            rho(nr,n)=rho(nr,n)+z1*xci(nr,n,index2)
          endif
        enddo
      enddo
    enddo
    coef=(-1d0)**k*s(k,km,kr)*s(k,kb,kn)
    ! w(point,r) = sum_n potential(point,n) rho(r,n).
    call dgemm('N','T',nhf,cr,cn,1d0,potential(1,nnn),nhf, &
               rho(nnr,nnn),nxx,0d0,w(1,nnr),nhf)
    v=0d0
    do nr=nnr,nmaxx(ir)
      v=v+phi(:,nr,ir)*w(:,nr)
    enddo
    ! Full weighted node projection, without applying quadrature twice.
    do m=nnm,nmaxx(im)
      rmo(m,i)=rmo(m,i)+coef*zz*sum(v*phi(:,m,im))
    enddo
  enddo
end subroutine
