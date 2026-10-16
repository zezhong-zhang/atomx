! Isolated full scalar Fock correction; bare F_mv is outside correlation Sigma1.
module native_valence_fock_control
  use iso_c_binding
  use native_sd_bridge, only: fock,offdiag,alpha,beta,rho_value
  use native_valence_control
  implicit none
contains
  subroutine atomx_val_linear_fock(energy,s,d,ds,dd,rs,rd,drs,drd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh),ds(nx),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    real(c_double) fs(nx),fd(nxx,nxx,nvh),dfs(nx),dfd(nxx,nxx,nvh)
    ! Exact homogeneous part: original core-only diagrams and bare source
    ! are absent. No subtraction of two nearly equal affine evaluations.
    rvo=s; xvo=d; drvo=ds; dxvo=dd
    call pci_val_outin
    call pci_val_terma4(energy)
    call pci_val_term2(energy)
    call pci_val_term41(energy)
    rs=rvo; rd=xvo; drs=drvo; drd=dxvo
    call valence_fock_map(energy,d,dd,fs,fd,dfs,dfd,include_core=.false.)
    rs=rs+fs; rd=rd+fd; drs=drs+dfs; drd=drd+dfd
  end subroutine

  subroutine atomx_val_map_full_fock(energy,s,d,ds,dd,rs,rd,drs,drd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh),ds(nx),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    real(c_double) fs(nx),fd(nxx,nxx,nvh),dfs(nx),dfd(nxx,nxx,nvh)
    call atomx_val_map(energy,s,d,ds,dd,rs,rd,drs,drd)
    call valence_fock_map(energy,d,dd,fs,fd,dfs,dfd)
    rs=rs+fs; rd=rd+fd; drs=drs+dfs; drd=drd+dfd
  end subroutine

  subroutine atomx_val_fock_map(energy,d,dd,rs,rd,drs,drd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    call valence_fock_map(energy,d,dd,rs,rd,drs,drd)
  end subroutine

  subroutine valence_fock_map(energy,d,dd,rs,rd,drs,drd,include_core)
    real(c_double),intent(in) :: energy,d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    logical,optional,intent(in) :: include_core
    logical core_terms
    real(c_double) t(nxx,nxx,nvh),dt(nxx,nxx,nvh),den,z,core_t,core_dt
    integer channel,b,im,jn,k,m,n,c,ic,jb,key,r,a1,a2
    core_terms=.true.
    if(present(include_core)) core_terms=include_core
    t=0; dt=0; rs=0; rd=0; drs=0; drd=0
    do channel=1,ivchan
      b=ivb(channel); im=ivm(channel); jn=ivn(channel)
      do n=first_virtual(jn),nmaxx(jn)
        do m=first_virtual(im),nmaxx(im)
          den=energy+eo(b)-ee(m,im)-ee(n,jn)
          t(m,n,channel)=d(m,n,channel)/den
          dt(m,n,channel)=dd(m,n,channel)/den-d(m,n,channel)/den**2
        enddo
      enddo
    enddo
    do channel=1,ivchan
      b=ivb(channel); im=ivm(channel); jn=ivn(channel); k=ivl(channel)
      jb=core_partial(b)
      rd(:,:,channel)=matmul(offdiag(:nxx,:nxx,im),t(:,:,channel)) &
                     +matmul(t(:,:,channel),transpose(offdiag(:nxx,:nxx,jn)))
      drd(:,:,channel)=matmul(offdiag(:nxx,:nxx,im),dt(:,:,channel)) &
                      +matmul(dt(:,:,channel),transpose(offdiag(:nxx,:nxx,jn)))
      do c=1,nn
        ic=core_partial(c)
        key=ipxv(c,im,jn,k)
        if(ic==jb.and.c/=b.and.key>0) then
          z=fock(core_slot(c),core_slot(b),ic)
          rd(:,:,channel)=rd(:,:,channel)-z*t(:,:,key)
          drd(:,:,channel)=drd(:,:,channel)-z*dt(:,:,key)
        endif
        if(.not.core_terms.or.ic/=vpartial) cycle
        z=fock(core_slot(c),vslot,ic)
        do n=first_virtual(jn),nmaxx(jn)
          do m=first_virtual(im),nmaxx(im)
            den=eo(c)+eo(b)-ee(m,im)-ee(n,jn)+energy-ev
            core_t=rho_value(m,n,c,b,im,jn,k)/den
            core_dt=-core_t/den
            rd(m,n,channel)=rd(m,n,channel)-z*core_t
            drd(m,n,channel)=drd(m,n,channel)-z*core_dt
          enddo
        enddo
      enddo
      rd(:first_virtual(im)-1,:,channel)=0; drd(:first_virtual(im)-1,:,channel)=0
      rd(:,:first_virtual(jn)-1,channel)=0; drd(:,:first_virtual(jn)-1,channel)=0
      if(nmaxx(im)<nxx) then
        rd(nmaxx(im)+1:,:,channel)=0; drd(nmaxx(im)+1:,:,channel)=0
      endif
      if(nmaxx(jn)<nxx) then
        rd(:,nmaxx(jn)+1:,channel)=0; drd(:,nmaxx(jn)+1:,channel)=0
      endif
    enddo
    im=vpartial
    do m=first_virtual(im),nmaxx(im)
      do c=1,nn
        ic=core_partial(c)
        if(core_terms.and.ic==im) then
          den=eo(c)-ee(m,im)+energy-ev
          z=fock(core_slot(c),vslot,ic)
          rs(m)=rs(m)-z*rmi(m,c)/den
          drs(m)=drs(m)+z*rmi(m,c)/den**2
        endif
        do r=first_virtual(ic),nmaxx(ic)
          z=fock(core_slot(c),r,ic)
          do k=0,kk
            a1=ipxv(c,im,ic,k); a2=ipxv(c,ic,im,k)
            if(a1>0) then
              rs(m)=rs(m)+z*alpha(im,ic,k)*t(m,r,a1)
              drs(m)=drs(m)+z*alpha(im,ic,k)*dt(m,r,a1)
            endif
            if(a2>0) then
              rs(m)=rs(m)-z*beta(im,ic,k)*t(r,m,a2)
              drs(m)=drs(m)-z*beta(im,ic,k)*dt(r,m,a2)
            endif
          enddo
        enddo
      enddo
    enddo
  end subroutine
end module
