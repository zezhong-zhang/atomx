! Native component-tagged specialization of the pCI GPL-3.0 term21.
! The pinned original source and license are retained by the external build.
! Each core-pair output channel belongs to exactly one worker. No reductions,
! density cutoffs, virtual truncation or changed angular equations are used.
subroutine term21_compact()
  use native_sd_bridge, only: nhf,nxx,nk,kk,ncore
  implicit none
  integer it,i,j,c,pairs,a(ncore*(ncore+1)/2),b(ncore*(ncore+1)/2)
  real(8),allocatable :: uu(:,:,:),acc(:,:,:,:),v1(:,:),uv(:,:),v(:),u(:),vv(:)
  pairs=0
  do i=1,ncore
    do j=1,i
      pairs=pairs+1; a(pairs)=i; b(pairs)=j
    enddo
  enddo
  !$omp parallel num_threads(CASE_THREADS) default(shared) private(uu,acc,v1,uv,v,u,vv,it)
  allocate(uu(nhf,nxx,nxx),acc(nhf,nxx,0:kk,nk),v1(nhf,nxx),uv(nhf,nxx),v(nhf),u(nhf),vv(nhf))
  !$omp do schedule(dynamic,1)
  do it=1,pairs
    call term21_compact_pair(a(it),b(it),uu,acc,v1,uv,v,u,vv)
  enddo
  !$omp end do
  deallocate(uu,acc,v1,uv,v,u,vv)
  !$omp end parallel
end subroutine

recursive subroutine term21_compact_pair(i,j,uu,acc,v1,uv,v,u,vv)
  use native_sd_bridge
  implicit none
  integer,intent(in) :: i,j
  real(c_double),intent(inout) :: uu(nhf,nxx,nxx),acc(nhf,nxx,0:kk,nk)
  real(c_double),intent(inout) :: v1(nhf,nxx),uv(nhf,nxx),v(nhf),u(nhf),vv(nhf)
  integer ka,kb,kapa,kapb,la,lb,ja,jb,inda,indb,n0a,n0b
  integer im,ir,iss,inn,km,kr,ks,kn,kapm,kapr,kaps,kapn,lm,lr,ls,ln,jm,jr,js,jn
  integer m0,n0r,n0s,n0n,nnm,nnr,nns,nnn,l,lmin,lmaxv,k1,k1min,k1max,k,kmin,kmax
  integer m,n,nr,nss,index1,index,c,point
  real(c_double) coef,z1,c1,cfin
  real(c_double),external :: s,d6j
  ka=ko(i); kb=ko(j)
  call klj(ka,kapa,la,ja,inda,n0a)
  call klj(kb,kapb,lb,jb,indb,n0b)
  do im=1,2*lmax+1
    call indk1(im,km,kapm,lm,jm,m0)
    call st(km,nnm)
    acc=0d0
    do ir=1,2*lmax+1
      call indk1(ir,kr,kapr,lr,jr,n0r)
      call st(kr,nnr)
      lmin=abs(kapm-kapr); lmaxv=kapm+kapr-1
      do l=lmin,lmaxv
        if(mod(lm+lr+l,2)/=0) cycle
        do m=nnm,nmaxx(im)
          do nr=nnr,nmaxx(ir)
            ! GG and FF occupy distinct auxiliary-node tags, including KB.
            v=(gg(:,nr,ir)+ff(:,nr,ir))*(gg(:,m,im)+ff(:,m,im))
            call density_action(v,u,l)
            uu(:,nr,m)=u
          enddo
        enddo
        do iss=1,2*lmax+1
          call indk1(iss,ks,kaps,ls,js,n0s)
          call st(ks,nns)
          if(mod(lr+la,2)/=mod(ls+lb,2)) cycle
          k1min=max(abs(kapr-kapa),abs(kaps-kapb))
          k1max=min(kapr+kapa-1,kaps+kapb-1)
          do k1=k1min,k1max
            index1=ipxc(i,j,ir,iss,k1)
            if(index1==0) error stop 'complete term21 source channel absent'
            do nr=nnr,nmaxx(ir)
              v1(:,nr)=0d0
              do nss=nns,nmaxx(iss)
                v1(:,nr)=v1(:,nr)+(gg(:,nss,iss)+ff(:,nss,iss))*xci(nr,nss,index1)
              enddo
            enddo
            do m=nnm,nmaxx(im)
              uv(:,m)=0d0
              do nr=nnr,nmaxx(ir)
                uv(:,m)=uv(:,m)+v1(:,nr)*uu(:,nr,m)
              enddo
            enddo
            do inn=1,2*lmax+1
              call indk1(inn,kn,kapn,ln,jn,n0n)
              call st(kn,nnn)
              if(mod(la+lm,2)/=mod(ln+lb,2)) cycle
              if(l<abs(kapn-kaps).or.l>kapn+kaps-1.or.mod(ln+ls+l,2)/=0) cycle
              coef=(-1d0)**l*s(l,km,kr)*s(l,kn,ks)
              z1=(-1d0)**(kapa+kapb+kapn+kapm)
              kmin=max(abs(kapm-kapa),abs(kapn-kapb),abs(k1-l))
              kmax=min(kapm+kapa-1,kapn+kapb-1,k1+l)
              do k=kmin,kmax
                c1=(2*k+1d0)*d6j(jm,ja,2*k,2*k1,2*l,jr)
                cfin=coef*c1*z1*d6j(jn,jb,2*k,2*k1,2*l,js)
                do m=nnm,nmaxx(im)
                  acc(:,m,k,inn)=acc(:,m,k,inn)+uv(:,m)*cfin
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    do inn=1,2*lmax+1
      call indk1(inn,kn,kapn,ln,jn,n0n)
      call st(kn,nnn)
      if(mod(la+lm,2)/=mod(ln+lb,2)) cycle
      kmin=max(abs(kapm-kapa),abs(kapn-kapb))
      kmax=min(kapm+kapa-1,kapn+kapb-1)
      do k=kmin,kmax
        index=ipxc(i,j,im,inn,k)
        if(index==0) error stop 'complete term21 target channel absent'
        do n=nnn,nmaxx(inn)
          do m=nnm,nmaxx(im)
            vv=acc(:,m,k,inn)*(gg(:,n,inn)+ff(:,n,inn))
            xco(m,n,index)=xco(m,n,index)+sum(vv)
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine
