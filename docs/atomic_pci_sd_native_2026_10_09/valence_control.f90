! Isolated pinned pCI GPL-3.0 valence convention control, native radial services.
module native_valence_control
  use iso_c_binding
  use native_sd_bridge, only: nhf,nx,nxx,nk,nn,kk,nch,nxv,nvh,kx1, &
    gg,ff,ee,eo,ko,no,core_partial,core_slot,first_virtual,nmaxx, &
    xci,rmi,ipa,ipb,ipm,ipn,ipl,icchan,vector_unpack,xco,rmo
  implicit none
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
