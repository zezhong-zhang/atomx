! Native fixed-energy CI+all-order SD bridge; pinned GPL-3.0 diagrams external.
! Full scalar Fock and exact homogeneous map independently anchored.
module native_valence_bridge
  use iso_c_binding
  use native_sd_bridge, only: nhf,nx,nxx,nk,nn,kk,nch,kx1, &
    gg,ff,ee,eo,ko,no,core_partial,core_slot,first_virtual,nmaxx, &
    xci,rmi,ipa,ipb,ipm,ipn,ipl,icchan,vector_unpack,xco,rmo
  implicit none
  ! Optimized core objects are compiled before valence dimensions are bound.
  ! Their exported NXV/NVH are placeholders and cannot define this COMMON ABI.
  include 'valence_control_sizes.inc'
  real(c_double) gv(nhf),fv(nhf),ev
  integer vp,vk
  common /val/ gv,fv,ev,vp,vk
  integer ipxv1(nn,nk,nk,0:kk),ipxv2(nn,nk,nk,0:kk),ipxv(nn,nk,nk,0:kk),ivchan
  real(c_double) xv1(nxx,nxx,nxv),xv2(nxx,nxx,nxv)
  common /xval1/ ipxv1,xv1
  common /xval2/ ipxv2,xv2
  common /irhov/ ipxv,ivchan
  real(c_double) xvi(nxx,nxx,nvh),rvi(nx),xvo(nxx,nxx,nvh),rvo(nx)
  real(c_double) dxvi(nxx,nxx,nvh),drvi(nx),dxvo(nxx,nxx,nvh),drvo(nx)
  common /rhovin/ xvi,rvi
  common /rhovout/ xvo,rvo
  common /drhovin/ dxvi,drvi
  common /drhovout/ dxvo,drvo
  integer ivb(nvh),ivm(nvh),ivn(nvh),ivl(nvh),vslot,vpartial
  common /ipv/ ivb,ivm,ivn,ivl
  real(c_double) val_source(nxx,nxx,nvh)
contains
  subroutine atomx_val_init(slot,kappa,total,t) bind(C)
    integer(c_int),value,intent(in) :: slot,kappa
    integer(c_int64_t),value,intent(in) :: total
    real(c_double),intent(in) :: t(total)
    integer kap,l,j,ind,n0,i,c,m,n,im,jn,key,iv
    call klj(kappa,kap,l,j,ind,n0)
    vslot=slot; vpartial=ind; vp=slot+n0; vk=kappa
    gv=gg(:,slot,ind); fv=ff(:,slot,ind); ev=ee(slot,ind)
    ipxv1=0; ipxv2=0; ipxv=0
    ivb=0; ivm=0; ivn=0; ivl=0; rvo=0; drvo=0
    call pci_val_xval
    key=0
    call pci_val_rhoval(key,iv)
    val_source=xvo
    ! CoreSD exports physical amplitudes. Original valsd-ci rhocore explicitly
    ! converts those amplitudes to sigmas before any shifted denominator.
    call vector_unpack(total,t)
    xci=xco; rmi=rmo
    do i=1,nn
      im=core_partial(i)
      do m=first_virtual(im),nmaxx(im)
        rmi(m,i)=rmi(m,i)*(eo(i)-ee(m,im))
      enddo
    enddo
    do c=1,icchan
      im=ipm(c); jn=ipn(c)
      do n=first_virtual(jn),nmaxx(jn)
        do m=first_virtual(im),nmaxx(im)
          xci(m,n,c)=xci(m,n,c)*(eo(ipa(c))+eo(ipb(c))-ee(m,im)-ee(n,jn))
        enddo
      enddo
    enddo
  end subroutine

  subroutine atomx_val_count(count) bind(C)
    integer(c_int),intent(out) :: count
    count=ivchan
  end subroutine

  subroutine atomx_val_info(channels,source,energy) bind(C)
    integer(c_int),intent(out) :: channels(4,nvh)
    real(c_double),intent(out) :: source(nxx,nxx,nvh),energy
    channels(1,:)=ivb; channels(2,:)=ivm; channels(3,:)=ivn; channels(4,:)=ivl
    source=val_source; energy=ev
  end subroutine

  subroutine atomx_val_map(energy,s,d,ds,dd,rs,rd,drs,drd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh),ds(nx),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    rvo=s; xvo=d; drvo=ds; dxvo=dd
    call pci_val_outin
    call pci_val_terma1(energy)
    call pci_val_terma2(energy)
    call pci_val_terma3(energy)
    call pci_val_terma4(energy)
    call pci_val_term1(energy)
    call pci_val_term2(energy)
    ! term31 is intentionally absent in the pinned CI+all-order main loop.
    call pci_val_term32(energy)
    call pci_val_term41(energy)
    call pci_val_term42(energy)
    call pci_val_term51(energy)
    call pci_val_term52(energy)
    rs=rvo; rd=xvo+val_source; drs=drvo; drd=dxvo
  end subroutine
  subroutine atomx_val_term(which,energy,d,result) bind(C)
    integer(c_int),value,intent(in) :: which
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: d(nxx,nxx,nvh)
    real(c_double),intent(out) :: result(nxx,nxx,nvh)
    rvo=0; xvo=d; drvo=0; dxvo=0
    call pci_val_outin
    select case(which)
    case(0)
      xvo=val_source
    case(1)
      call pci_val_term1(energy)
    case(2)
      call pci_val_term2(energy)
    case(3)
      call pci_val_term32(energy)
    case(4)
      call pci_val_term41(energy)
    case(5)
      call pci_val_term42(energy)
    case(6)
      call pci_val_term51(energy)
    case(7)
      call pci_val_term52(energy)
    end select
    result=xvo
  end subroutine
end module
! Complete scalar Fock correction; bare F_mv is outside correlation Sigma1.
module native_valence_fock_bridge
  use iso_c_binding
  use native_sd_bridge, only: fock,offdiag,alpha,beta,rho_value
  use native_valence_bridge
  implicit none
contains
  subroutine atomx_val_linear_fock(energy,s,d,ds,dd,rs,rd,drs,drd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh),ds(nx),dd(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh),drs(nx),drd(nxx,nxx,nvh)
    rvo=s; xvo=d; drvo=ds; dxvo=dd
    call valence_linear_action(energy,.true.)
    rs=rvo; rd=xvo; drs=drvo; drd=dxvo
  end subroutine

  subroutine atomx_val_linear_value_fock(energy,s,d,rs,rd) bind(C)
    real(c_double),value,intent(in) :: energy
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh)
    rvo=s; xvo=d; drvo=0d0; dxvo=0d0
    call valence_linear_action(energy,.false.)
    ! No derivative output is offered by this value-only entry point.
    rs=rvo; rd=xvo
  end subroutine

  subroutine valence_linear_action(energy,calculate_response)
    real(c_double),intent(in) :: energy
    logical,intent(in) :: calculate_response
    real(c_double) fs(nx),fd(nxx,nxx,nvh),dfs(nx),dfd(nxx,nxx,nvh)
    ! Exact homogeneous part: original core-only diagrams and bare source
    ! are absent. No subtraction of two nearly equal affine evaluations.
    call pci_val_outin
    call pci_val_terma4(energy)
    if(calculate_response) then
      call pci_val_term2(energy)
    else
      call pci_val_term2(energy) ! value-only policy
    endif
    call pci_val_term41(energy)
    ! OUTIN binds the complete input numerators; homogeneous diagrams read
    ! them without mutation. Reuse that exact input for the shared Fock map.
    call valence_fock_map(energy,xvi,dxvi,fs,fd,dfs,dfd,include_core=.false.)
    rvo=rvo+fs; xvo=xvo+fd; drvo=drvo+dfs; dxvo=dxvo+dfd
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
