! Exact component-tagged specialization of pinned pCI GPL-3.0 term31/term32.
! Shared full core-single trial densities and weighted Coulomb potentials;
! each worker writes disjoint core-pair channels. No amplitude screening.
subroutine core_single_to_doubles_blas(which)
  use native_sd_bridge
  implicit none
  integer,intent(in) :: which
  integer i,j,q,other,partial,k,ka,kap,l,j2,n0,kmin,kmax,a,first,last
  real(c_double),allocatable :: phi(:,:,:),trial(:,:),potential(:,:),w(:,:),v(:),u(:)
  if(which/=31.and.which/=32) error stop 'unsupported single-to-double diagram'
  allocate(phi(nhf,nxx,nk),trial(nhf,nn),potential(nhf,nxx))
  phi=gg(:,:nxx,:nk)+ff(:,:nxx,:nk)
  trial=0d0
  ! Native rmi is already the complete physical core single amplitude.
  do i=1,ncore
    partial=core_partial(i); first=first_virtual(partial); last=nmaxx(partial)
    if(last>=first) call dgemv('N',nhf,last-first+1,1d0,phi(:,first,partial),nhf, &
                              rmi(first,i),1,0d0,trial(:,i),1)
  enddo
  !$omp parallel num_threads(CASE_THREADS) default(shared) &
  !$omp private(i,j,q,other,partial,k,ka,kap,l,j2,n0,kmin,kmax,a,first,last,w,v,u)
  allocate(w(nhf,nxx),v(nhf),u(nhf))
  do q=1,ncore
    ka=ko(q)
    call klj(ka,kap,l,j2,partial,n0)
    do partial=1,nk
      call indk1(partial,ka,kap,l,j2,n0)
      kmin=abs(kap-abs(ko(q))); kmax=kap+abs(ko(q))-1
      first=first_virtual(partial); last=nmaxx(partial)
      do k=kmin,kmax
        if(mod(l+(merge(ko(q),-ko(q)-1,ko(q)>0))+k,2)/=0) cycle
        !$omp do schedule(static)
        do a=first,last
          v=(go(:,q)+fo(:,q))*phi(:,a,partial)
          call density_action(v,u,k)
          potential(:,a)=u
        enddo
        !$omp end do
        !$omp do schedule(dynamic,1)
        do other=1,ncore
          if(which==31) then
            i=other; j=q
          else
            i=q; j=other
          endif
          if(i<j) cycle
          call single_to_doubles_pair(which,i,j,partial,k,phi,trial,potential,w)
        enddo
        !$omp end do
      enddo
    enddo
  enddo
  deallocate(w,v,u)
  !$omp end parallel
  deallocate(phi,trial,potential)
end subroutine

recursive subroutine single_to_doubles_pair(which,i,j,partial,k,phi,trial,potential,w)
  use native_sd_bridge
  implicit none
  integer,intent(in) :: which,i,j,partial,k
  real(c_double),intent(in) :: phi(nhf,nxx,nk),trial(nhf,nn),potential(nhf,nxx)
  real(c_double),intent(inout) :: w(nhf,nxx)
  integer im,jn,km,kn,kapm,kapn,lm,ln,jm,jnn,m0,n0,m,n,index,firstm,firstn,cm,cn
  real(c_double) c
  real(c_double),external :: s
  do m=1,nk
    if(which==31) then
      im=m; jn=partial
    else
      im=partial; jn=m
    endif
    call indk1(im,km,kapm,lm,jm,m0)
    call indk1(jn,kn,kapn,ln,jnn,n0)
    index=ipxc(i,j,im,jn,k)
    if(index==0) cycle
    ! ipxc also contains channels of odd electric parity. Exactly reject
    ! those channels using the original triangle/parity rules, not magnitude.
    if(mod(lm+merge(ko(i),-ko(i)-1,ko(i)>0)+k,2)/=0) cycle
    if(mod(ln+merge(ko(j),-ko(j)-1,ko(j)>0)+k,2)/=0) cycle
    if(k<abs(kapm-abs(ko(i))).or.k>kapm+abs(ko(i))-1) cycle
    if(k<abs(kapn-abs(ko(j))).or.k>kapn+abs(ko(j))-1) cycle
    firstm=first_virtual(im); firstn=first_virtual(jn)
    cm=nmaxx(im)-firstm+1; cn=nmaxx(jn)-firstn+1
    if(cm<=0.or.cn<=0) cycle
    c=(-1d0)**k*s(k,km,ko(i))*s(k,kn,ko(j))
    if(which==31) then
      do n=firstm,nmaxx(im)
        w(:,n)=trial(:,i)*phi(:,n,im)
      enddo
      call dgemm('T','N',cm,cn,nhf,c,w(:,firstm),nhf,potential(:,firstn),nhf, &
                  1d0,xco(firstm,firstn,index),nxx)
    else
      do n=firstn,nmaxx(jn)
        w(:,n)=trial(:,j)*phi(:,n,jn)
      enddo
      call dgemm('T','N',cm,cn,nhf,c,potential(:,firstm),nhf,w(:,firstn),nhf, &
                  1d0,xco(firstm,firstn,index),nxx)
    endif
  enddo
end subroutine
