! Isolated irreducible two-valence scalar Fov contribution.
module native_sigma2_fock_control
  use iso_c_binding
  use native_sd_bridge, only: fock,eo,core_slot,core_partial,first_virtual,nmaxx
  use native_sigma2_control
  implicit none
contains
  subroutine atomx_vw_fock(count,channels,result,derivative) bind(C)
    integer(c_int),intent(out) :: count,channels(5,nvwh)
    real(c_double),intent(out) :: result(nxx,nxx,0:nvwh),derivative(nxx,nxx,0:nvwh)
    real(c_double) z,den,c0,pci_vw_s1
    integer channel,v,w,im,jn,k,c,m,n,ic,ivp,iwp,kap,l,j,n0v,n0w,a,b
    call pci_vw_ini
    count=ichan; channels=0; result=0; derivative=0
    channels(1,:ichan)=ov(:ichan); channels(2,:ichan)=ow(:ichan)
    channels(3,:ichan)=om(:ichan); channels(4,:ichan)=on(:ichan)
    channels(5,:ichan)=ol(:ichan)
    do channel=1,ichan
      v=ov(channel); w=ow(channel); im=om(channel); jn=on(channel); k=ol(channel)
      call klj(vk(v),kap,l,j,ivp,n0v)
      call klj(vk(w),kap,l,j,iwp,n0w)
      call indk1(im,a,kap,l,j,n0v)
      call indk1(jn,b,kap,l,j,n0w)
      c0=(-1d0)**k*pci_vw_s1(k,a,vk(v))*pci_vw_s1(k,b,vk(w))
      if(c0==0d0) error stop 'zero complete screening radial normalization'
      do c=1,nn
        ic=core_partial(c)
        a=ipxv(v,c,im,jn,k); b=ipxv(w,c,jn,im,k)
        do n=first_virtual(jn),nmaxx(jn)
          do m=first_virtual(im),nmaxx(im)
            if(ic==iwp.and.a>0) then
              call klj(vk(w),kap,l,j,iwp,n0w)
              z=-fock(core_slot(c),vp(w)-n0w,ic)/c0
              den=evt(v)+evt(w)+eo(c)-ev(w)-ee(m,im)-ee(n,jn)
              result(m,n,channel)=result(m,n,channel)+z*xvi(m,n,a)/den
              derivative(m,n,channel)=derivative(m,n,channel) &
                +z*(dxvi(m,n,a)/den-2*xvi(m,n,a)/den**2)
            endif
            if(ic==ivp.and.b>0) then
              call klj(vk(v),kap,l,j,ivp,n0v)
              z=-fock(core_slot(c),vp(v)-n0v,ic)/c0
              den=evt(v)+evt(w)+eo(c)-ev(v)-ee(m,im)-ee(n,jn)
              result(m,n,channel)=result(m,n,channel)+z*xvi(n,m,b)/den
              derivative(m,n,channel)=derivative(m,n,channel) &
                +z*(dxvi(n,m,b)/den-2*xvi(n,m,b)/den**2)
            endif
          enddo
        enddo
      enddo
    enddo
  end subroutine
end module
