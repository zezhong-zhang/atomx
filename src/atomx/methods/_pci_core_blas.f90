! Exact BLAS construction of the original pCI X1/X2 channels.
! Included inside native_sd_bridge; density_action is the shared native
! weighted cell-Coulomb service. Original diagrams remain independent.
  subroutine rhocore_blas()
    integer i,j,im,jn,km,kn,kapm,kapn,lm,ln,jm,jnn,m0,n0
    integer ka,kb,kapa,kapb,la,lb,ja,jb,inda,indb,n0a,n0b,k,lo,hi,m,nm,nnv
    real(c_double) c
    real(c_double), external :: s
    real(c_double) left(nhf,nxx),right(nhf,nxx),acted(nhf,nxx),v(nhf),u(nhf)
    ichan1=0; ichan2=0; ipx1=0; ipx2=0; x1=0d0; x2=0d0
    do i=1,ncore
      ka=ko(i)
      call klj(ka,kapa,la,ja,inda,n0a)
      do j=1,i
        kb=ko(j)
        call klj(kb,kapb,lb,jb,indb,n0b)
        do im=1,2*lmax+1
          call indk1(im,km,kapm,lm,jm,m0)
          nm=nmaxx(im)
          do jn=1,2*lmax+1
            call indk1(jn,kn,kapn,ln,jnn,n0)
            nnv=nmaxx(jn)
            lo=max(abs(kapm-kapa),abs(kapn-kapb))
            hi=min(kapm+kapa-1,kapn+kapb-1)
            do k=lo,hi
              if(mod(lm+la+k,2)/=0.or.mod(ln+lb+k,2)/=0) cycle
              ichan1=ichan1+1
              if(ichan1>nxk) error stop 'complete X1 channel resource exceeded'
              ipx1(i,j,im,jn,k)=ichan1
              c=(-1d0)**k*s(k,km,ka)*s(k,kn,kb)
              do m=1,nm
                left(:,m)=go(:,i)*gg(:,m,im)+fo(:,i)*ff(:,m,im)
                call density_action(left(:,m),acted(:,m),k)
              enddo
              do m=1,nnv
                right(:,m)=go(:,j)*gg(:,m,jn)+fo(:,j)*ff(:,m,jn)
              enddo
              if(nm>0.and.nnv>0) then
                call dgemm('T','N',nm,nnv,nhf,c,acted,nhf,right,nhf,0d0,x1(:,:,ichan1),nxx)
              endif
            enddo
            lo=max(abs(kapm-kapn),abs(kapa-kapb))
            hi=min(kapm+kapn-1,kapa+kapb-1)
            do k=lo,hi
              if(mod(lm+ln+k,2)/=0.or.mod(la+lb+k,2)/=0) cycle
              ichan2=ichan2+1
              if(ichan2>nxk) error stop 'complete X2 channel resource exceeded'
              ipx2(i,j,im,jn,k)=ichan2
              c=(-1d0)**k*s(k,km,kn)*s(k,ka,kb)
              v=go(:,i)*go(:,j)+fo(:,i)*fo(:,j)
              call density_action(v,u,k)
              ! P/Q samples have disjoint component tags. GG+FF therefore
              ! combines their complete densities without a cross component.
              do m=1,nm
                left(:,m)=(gg(:,m,im)+ff(:,m,im))*u
              enddo
              do m=1,nnv
                right(:,m)=gg(:,m,jn)+ff(:,m,jn)
              enddo
              if(nm>0.and.nnv>0) then
                call dgemm('T','N',nm,nnv,nhf,c,left,nhf,right,nhf,0d0,x2(:,:,ichan2),nxx)
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  end subroutine
