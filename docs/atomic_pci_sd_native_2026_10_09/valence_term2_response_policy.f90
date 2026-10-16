! Exact native specialization of pinned pCI GPL-3.0 valsd-ci term2.
! Parallel exact shared complete virtual-pair potential; signed angular contractions and
! analytic total energy derivative retained. No amplitude screening.
subroutine valence_term2_streamed(energy,calculate_response)
  use native_sd_bridge, only: density_action
  use native_valence_bridge
  implicit none
  real(c_double),intent(in) :: energy
  logical,intent(in) :: calculate_response
  real(c_double),allocatable :: phi(:,:,:),uu(:,:,:),v(:),u(:)
  real(c_double),allocatable :: rho(:,:),drho(:,:),v1(:,:),dv1(:,:),uv(:,:),duv(:,:)
  real(c_double),allocatable :: projected(:,:),dprojected(:,:)
  integer im,ir,j,m,nr,km,kr,kapm,kapr,lm,lr,jm,jr,m0,r0,nnm,nnr
  integer l,lmin,lhigh
  allocate(phi(nhf,nx,nk),uu(nhf,nxx,nxx))
  phi=gg+ff
  !$omp parallel num_threads(CASE_THREADS) default(shared) &
  !$omp private(v,u,rho,drho,v1,dv1,uv,duv,projected,dprojected,im,ir,j,m,nr,km,kr,kapm,kapr,lm,lr,jm,jr,m0,r0,nnm,nnr,l,lmin,lhigh)
  allocate(v(nhf),u(nhf))
  allocate(rho(nxx,nxx),drho(nxx,nxx),v1(nhf,nxx),dv1(nhf,nxx))
  allocate(uv(nhf,nxx),duv(nhf,nxx),projected(nxx,nxx),dprojected(nxx,nxx))
  ! GG/FF are disjoint tagged native component coordinates; RP=1 and the
  ! original rint sums these coordinates without an extra quadrature weight.
  do im=1,nk
    call indk1(im,km,kapm,lm,jm,m0)
    call st(km,nnm)
    if(nmaxx(im)<nnm) cycle
    do ir=1,nk
      call indk1(ir,kr,kapr,lr,jr,r0)
      call st(kr,nnr)
      if(nmaxx(ir)<nnr) cycle
      lmin=abs(kapm-kapr); lhigh=kapm+kapr-1
      do l=lmin,lhigh
        if(mod(lm+lr+l,2)/=0) cycle
        !$omp do collapse(2) schedule(static)
        do m=nnm,nmaxx(im)
          do nr=nnr,nmaxx(ir)
            v=phi(:,nr,ir)*phi(:,m,im)
            call density_action(v,u,l)
            uu(:,nr,m)=u
          enddo
        enddo
        !$omp end do
        !$omp do schedule(dynamic,1)
        do j=1,nn
          call valence_term2_streamed_core(energy,calculate_response,j,im,ir,l,phi,uu,rho,drho, &
            v1,dv1,uv,duv,projected,dprojected)
        enddo
        !$omp end do
      enddo
    enddo
  enddo
  deallocate(v,u,rho,drho,v1,dv1,uv,duv,projected,dprojected)
  !$omp end parallel
  deallocate(phi,uu)
end subroutine

recursive subroutine valence_term2_streamed_core(energy,calculate_response,j,im,ir,l,phi,uu,rho,drho, &
  v1,dv1,uv,duv,projected,dprojected)
  use native_valence_bridge
  implicit none
  real(c_double),intent(in) :: energy,phi(nhf,nx,nk),uu(nhf,nxx,nxx)
  integer,intent(in) :: j,im,ir,l
  logical,intent(in) :: calculate_response
  real(c_double),intent(inout) :: rho(nxx,nxx),drho(nxx,nxx)
  real(c_double),intent(inout) :: v1(nhf,nxx),dv1(nhf,nxx),uv(nhf,nxx),duv(nhf,nxx)
  real(c_double),intent(inout) :: projected(nxx,nxx),dprojected(nxx,nxx)
  integer ka,kb,kapa,kapb,la,lb,ja,jb,inda,indb,a0,b0
  integer km,kr,ks,kn,kapm,kapr,kaps,kapn,lm,lr,ls,ln,jm,jr,js,jn
  integer m0,r0,s0,n0,nnm,nnr,nns,nnn,iss,inn,k1,k,kmin,kmax,k1min,k1max
  integer index1,index,cm,cr,cs,cn,m,n,nr,nss
  real(c_double) coef,z1,c1,cfin,inverse,amplitude
  real(c_double),external :: s,d6j
  ka=vk; kb=ko(j)
  call klj(ka,kapa,la,ja,inda,a0)
  call klj(kb,kapb,lb,jb,indb,b0)
  call indk1(im,km,kapm,lm,jm,m0)
  call indk1(ir,kr,kapr,lr,jr,r0)
  call st(km,nnm); call st(kr,nnr)
  cm=nmaxx(im)-nnm+1; cr=nmaxx(ir)-nnr+1
  do iss=1,nk
    call indk1(iss,ks,kaps,ls,js,s0)
    call st(ks,nns)
    if(mod(lr+la,2)/=mod(ls+lb,2)) cycle
    cs=nmaxx(iss)-nns+1
    if(cs<=0) cycle
    k1min=max(abs(kapr-kapa),abs(kaps-kapb))
    k1max=min(kapr+kapa-1,kaps+kapb-1)
    do k1=k1min,k1max
      index1=ipxv(j,ir,iss,k1)
      if(index1==0) error stop 'complete streamed valence term2 source absent'
      do nss=nns,nmaxx(iss)
        do nr=nnr,nmaxx(ir)
          inverse=1d0/(energy+eo(j)-ee(nr,ir)-ee(nss,iss))
          amplitude=xvi(nr,nss,index1)*inverse
          rho(nr,nss)=amplitude
          if(calculate_response) drho(nr,nss)=dxvi(nr,nss,index1)*inverse-amplitude*inverse
        enddo
      enddo
      call dgemm('N','T',nhf,cr,cs,1d0,phi(1,nns,iss),nhf, &
        rho(nnr,nns),nxx,0d0,v1(1,nnr),nhf)
      if(calculate_response) then
      call dgemm('N','T',nhf,cr,cs,1d0,phi(1,nns,iss),nhf, &
        drho(nnr,nns),nxx,0d0,dv1(1,nnr),nhf)
      endif
      do m=nnm,nmaxx(im)
        uv(:,m)=0d0
        if(calculate_response) duv(:,m)=0d0
        do nr=nnr,nmaxx(ir)
          uv(:,m)=uv(:,m)+v1(:,nr)*uu(:,nr,m)
          if(calculate_response) duv(:,m)=duv(:,m)+dv1(:,nr)*uu(:,nr,m)
        enddo
      enddo
      do inn=1,nk
        call indk1(inn,kn,kapn,ln,jn,n0)
        call st(kn,nnn)
        if(mod(la+lm,2)/=mod(ln+lb,2)) cycle
        if(l<abs(kapn-kaps).or.l>kapn+kaps-1.or.mod(ln+ls+l,2)/=0) cycle
        cn=nmaxx(inn)-nnn+1
        if(cn<=0) cycle
        call dgemm('T','N',cm,cn,nhf,1d0,uv(1,nnm),nhf, &
          phi(1,nnn,inn),nhf,0d0,projected,nxx)
        if(calculate_response) then
        call dgemm('T','N',cm,cn,nhf,1d0,duv(1,nnm),nhf, &
          phi(1,nnn,inn),nhf,0d0,dprojected,nxx)
        endif
        coef=(-1d0)**l*s(l,km,kr)*s(l,kn,ks)
        z1=(-1d0)**(kapa+kapb+kapn+kapm)
        kmin=max(abs(kapm-kapa),abs(kapn-kapb),abs(k1-l))
        kmax=min(kapm+kapa-1,kapn+kapb-1,k1+l)
        if(kmax>kk) error stop 'complete valence term2 angular rank exceeded'
        do k=kmin,kmax
          c1=(2*k+1d0)*d6j(jm,ja,2*k,2*k1,2*l,jr)
          cfin=coef*c1*z1*d6j(jn,jb,2*k,2*k1,2*l,js)
          index=ipxv(j,im,inn,k)
          if(index==0) error stop 'complete streamed valence term2 target absent'
          do n=1,cn
            do m=1,cm
              xvo(m+nnm-1,n+nnn-1,index)=xvo(m+nnm-1,n+nnn-1,index)+projected(m,n)*cfin
              if(calculate_response) dxvo(m+nnm-1,n+nnn-1,index)=dxvo(m+nnm-1,n+nnn-1,index)+dprojected(m,n)*cfin
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine


subroutine atomx_val_term2_streamed_map(energy,d,dd,rd,pd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
  xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
  call valence_term2_streamed(energy,.true.)
  rd=xvo; pd=dxvo
end subroutine

subroutine atomx_val_term2_value_map(energy,d,rd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh)
  xvi=d; xvo=0d0
  call valence_term2_streamed(energy,.false.)
  rd=xvo
end subroutine

subroutine atomx_val_term2_original_map(energy,d,dd,rd,pd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
  xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
  call pci_val_term2(energy)
  rd=xvo; pd=dxvo
end subroutine
