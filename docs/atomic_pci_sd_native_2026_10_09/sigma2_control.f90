! Isolated original pCI two-valence screening, GPL-3.0 source retained.
module native_sigma2_control
  use iso_c_binding
  use native_sd_bridge, only: nhf,nx,nxx,nk,nn,kk,gg,ff,ee
  implicit none
  include 'vw_control_sizes.inc'
  real(c_double) gv(nhf,nz),fv(nhf,nz),ev(nz),evt(nz)
  integer vp(nz),vk(nz),nval
  common /vw_val/ gv,fv,ev,vp,vk,nval
  common /denergy/ evt
  integer ipvw(nz,nz,nk,nk,0:kk),ichan
  real(c_double) xvw(nxx,nxx,0:nvwh),dxvw(nxx,nxx,0:nvwh)
  common /radvw/ ipvw,xvw,ichan
  common /dradvw/ dxvw
  integer om(nvwh),on(nvwh),ov(nvwh),ow(nvwh),ol(nvwh)
  common /add/ om,on,ov,ow,ol
  integer ipxv(nz,nn,nk,nk,0:kk),ivchan
  real(c_double) xvi(nxx,nxx,0:nvh2),dxvi(nxx,nxx,0:nvh2),rsigma(nz,nx),drsigma(nz,nx)
  common /vw_irhov/ ipxv,ivchan
  common /rhov/ xvi,rsigma
  common /drhov/ dxvi,drsigma
  integer inmax
  common /x12ini/ inmax
contains
  subroutine atomx_vw_init(count,keys,energies) bind(C)
    integer(c_int),value,intent(in) :: count
    integer(c_int),intent(in) :: keys(2,count)
    real(c_double),intent(in) :: energies(count)
    integer i,kap,l,j,ind,n0
    if(count>nz.or.count<1) error stop 'complete valence count exceeds control cap'
    nval=count; inmax=nk
    gv=0; fv=0; ev=0; evt=0; vp=0; vk=0
    ipxv=0; xvi=0; dxvi=0; rsigma=0; drsigma=0; ivchan=0
    do i=1,count
      vp(i)=keys(1,i); vk(i)=keys(2,i)
      call klj(vk(i),kap,l,j,ind,n0)
      gv(:,i)=gg(:,vp(i)-n0,ind); fv(:,i)=ff(:,vp(i)-n0,ind)
      ev(i)=ee(vp(i)-n0,ind); evt(i)=energies(i)
    enddo
    call pci_vw_xcnvw
    call pci_vw_xfor4
  end subroutine

  subroutine atomx_vw_load(slot,count,channels,d,dd) bind(C)
    integer(c_int),value,intent(in) :: slot,count
    integer(c_int),intent(in) :: channels(4,count)
    real(c_double),intent(in) :: d(nxx,nxx,count),dd(nxx,nxx,count)
    integer i,c
    if(slot<1.or.slot>nval.or.ivchan+count>nvh2) error stop 'screening load exceeds control cap'
    do i=1,count
      ivchan=ivchan+1; c=ivchan
      ipxv(slot,channels(1,i),channels(2,i),channels(3,i),channels(4,i))=c
      xvi(:,:,c)=d(:,:,i); dxvi(:,:,c)=dd(:,:,i)
    enddo
  end subroutine

  subroutine atomx_vw_evaluate(count,channels,result,derivative) bind(C)
    integer(c_int),intent(out) :: count,channels(5,nvwh)
    real(c_double),intent(out) :: result(nxx,nxx,0:nvwh),derivative(nxx,nxx,0:nvwh)
    call pci_vw_ini
    call pci_vw_term1
    call pci_vw_term51
    call pci_vw_term52
    call pci_vw_term41a
    call pci_vw_term41c
    call pci_vw_term41d
    call pci_vw_term42a
    call pci_vw_term42c
    call pci_vw_term42d
    count=ichan; channels=0
    channels(1,:ichan)=ov(:ichan); channels(2,:ichan)=ow(:ichan)
    channels(3,:ichan)=om(:ichan); channels(4,:ichan)=on(:ichan); channels(5,:ichan)=ol(:ichan)
    result=xvw; derivative=dxvw
  end subroutine

  subroutine atomx_vw_evaluate_joint(count,channels,result,derivative) bind(C)
    ! Joint shift E_v -> E_v+delta for every column. Each explicit screening
    ! denominator contains two shifted valence energies. Original dxvw has
    ! one denominator derivative plus the supplied numerator responses.
    ! A second original call with zero numerator responses supplies the
    ! missing denominator derivative; no equation values are changed.
    integer(c_int),intent(out) :: count,channels(5,nvwh)
    real(c_double),intent(out) :: result(nxx,nxx,0:nvwh),derivative(nxx,nxx,0:nvwh)
    real(c_double) saved(nxx,nxx,0:nvh2),value2(nxx,nxx,0:nvwh),fixed(nxx,nxx,0:nvwh)
    saved=dxvi
    call atomx_vw_evaluate(count,channels,result,derivative)
    dxvi=0
    call atomx_vw_evaluate(count,channels,value2,fixed)
    dxvi=saved
    derivative=derivative+fixed
  end subroutine
end module
