! Exact native specialization of pinned pCI GPL-3.0 term21.
! Virtual pair potentials do not depend on the core pair. Generate each
! complete partial/rank block once, then stream every core-pair contraction.
! Component node tags are disjoint. No integral/amplitude cutoff is used.
subroutine term21_streamed()
  use native_sd_bridge
  implicit none
  integer im,ir,km,kr,kapm,kapr,lm,lr,jm,jr,m0,r0,nnm,nnr,l,lmin,lmaxv
  integer i,j,it,m,nr,pairs,a(nn*(nn+1)/2),b(nn*(nn+1)/2)
  real(c_double),allocatable :: phi(:,:,:),uu(:,:,:),v1(:,:),uv(:,:),projected(:,:)
  real(c_double),allocatable :: v(:),u(:)
  allocate(phi(nhf,nx,nk),uu(nhf,nxx,nxx))
  phi=gg+ff
  pairs=0
  do i=1,ncore
    do j=1,i
      pairs=pairs+1; a(pairs)=i; b(pairs)=j
    enddo
  enddo
  !$omp parallel num_threads(CASE_THREADS) default(shared) &
  !$omp private(v1,uv,projected,v,u,it,m,nr,im,ir,km,kr,kapm,kapr,lm,lr,jm,jr,m0,r0,nnm,nnr,l,lmin,lmaxv)
  allocate(v1(nhf,nxx),uv(nhf,nxx),projected(nxx,nxx),v(nhf),u(nhf))
  ! All workers encounter the same block loops; only disjoint density and
  ! core-pair outputs are distributed. The implicit barriers bind each block.
  do im=1,nk
    call indk1(im,km,kapm,lm,jm,m0)
    call st(km,nnm)
    do ir=1,nk
      call indk1(ir,kr,kapr,lr,jr,r0)
      call st(kr,nnr)
      lmin=abs(kapm-kapr); lmaxv=kapm+kapr-1
      do l=lmin,lmaxv
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
        do it=1,pairs
          call term21_streamed_pair(a(it),b(it),im,ir,l,phi,uu,v1,uv,projected)
        enddo
        !$omp end do
      enddo
    enddo
  enddo
  deallocate(v1,uv,projected,v,u)
  !$omp end parallel
  deallocate(phi,uu)
end subroutine

recursive subroutine term21_streamed_pair(i,j,im,ir,l,phi,uu,v1,uv,projected)
  use native_sd_bridge
  implicit none
  integer,intent(in) :: i,j,im,ir,l
  real(c_double),intent(in) :: phi(nhf,nx,nk),uu(nhf,nxx,nxx)
  real(c_double),intent(inout) :: v1(nhf,nxx),uv(nhf,nxx),projected(nxx,nxx)
  integer ka,kb,kapa,kapb,la,lb,ja,jb,inda,indb,n0a,n0b
  integer km,kr,ks,kn,kapm,kapr,kaps,kapn,lm,lr,ls,ln,jm,jr,js,jn
  integer m0,r0,s0,n0,nnm,nnr,nns,nnn,iss,inn,m,n,nr,k1,k,kmin,kmax,k1min,k1max
  integer index1,index,cm,cn,cr,cs
  real(c_double) coef,z1,c1,cfin
  real(c_double),external :: s,d6j
  ka=ko(i); kb=ko(j)
  call klj(ka,kapa,la,ja,inda,n0a)
  call klj(kb,kapb,lb,jb,indb,n0b)
  call indk1(im,km,kapm,lm,jm,m0)
  call indk1(ir,kr,kapr,lr,jr,r0)
  call st(km,nnm); call st(kr,nnr)
  cm=nmaxx(im)-nnm+1; cr=nmaxx(ir)-nnr+1
  do iss=1,nk
    call indk1(iss,ks,kaps,ls,js,s0)
    call st(ks,nns)
    if(mod(lr+la,2)/=mod(ls+lb,2)) cycle
    cs=nmaxx(iss)-nns+1
    k1min=max(abs(kapr-kapa),abs(kaps-kapb))
    k1max=min(kapr+kapa-1,kaps+kapb-1)
    do k1=k1min,k1max
      index1=ipxc(i,j,ir,iss,k1)
      if(index1==0) error stop 'complete streamed term21 source absent'
      if(cm==0.or.cr==0.or.cs==0) cycle
      ! V1(point,r) = sum_s phi_s(point) rho(r,s).
      call dgemm('N','T',nhf,cr,cs,1d0,phi(1,nns,iss),nhf, &
                 xci(nnr,nns,index1),nxx,0d0,v1(1,nnr),nhf)
      do m=nnm,nmaxx(im)
        uv(:,m)=0d0
        do nr=nnr,nmaxx(ir)
          uv(:,m)=uv(:,m)+v1(:,nr)*uu(:,nr,m)
        enddo
      enddo
      do inn=1,nk
        call indk1(inn,kn,kapn,ln,jn,n0)
        call st(kn,nnn)
        if(mod(la+lm,2)/=mod(ln+lb,2)) cycle
        if(l<abs(kapn-kaps).or.l>kapn+kaps-1.or.mod(ln+ls+l,2)/=0) cycle
        cn=nmaxx(inn)-nnn+1
        if(cn==0) cycle
        ! Sum the complete weighted auxiliary-node bilinear form once;
        ! each angular target rank multiplies the same radial matrix.
        call dgemm('T','N',cm,cn,nhf,1d0,uv(1,nnm),nhf, &
                   phi(1,nnn,inn),nhf,0d0,projected,nxx)
        coef=(-1d0)**l*s(l,km,kr)*s(l,kn,ks)
        z1=(-1d0)**(kapa+kapb+kapn+kapm)
        kmin=max(abs(kapm-kapa),abs(kapn-kapb),abs(k1-l))
        kmax=min(kapm+kapa-1,kapn+kapb-1,k1+l)
        do k=kmin,kmax
          c1=(2*k+1d0)*d6j(jm,ja,2*k,2*k1,2*l,jr)
          cfin=coef*c1*z1*d6j(jn,jb,2*k,2*k1,2*l,js)
          index=ipxc(i,j,im,inn,k)
          if(index==0) error stop 'complete streamed term21 target absent'
          do n=1,cn
            do m=1,cm
              xco(m+nnm-1,n+nnn-1,index)=xco(m+nnm-1,n+nnn-1,index)+projected(m,n)*cfin
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine
