! Standalone monovalent SD: physical singles, fixed-core denominators.
! External GPL-3.0 pinned pCI diagrams; serialized by the owned CoreSD lock.
module native_monovalent_bridge
  use iso_c_binding
  use native_valence_fock_bridge
  implicit none
contains
  subroutine atomx_monovalent_map(linear,s,d,rs,rd) bind(C)
    integer(c_int),value,intent(in) :: linear
    real(c_double),intent(in) :: s(nx),d(nxx,nxx,nvh)
    real(c_double),intent(out) :: rs(nx),rd(nxx,nxx,nvh)
    real(c_double) zero_s(nx),zero_d(nxx,nxx,nvh)
    real(c_double) prime_s(nx),prime_d(nxx,nxx,nvh)
    integer b,c,m
    zero_s=0d0; zero_d=0d0
    ! d holds D0 * physical rho; the original inverse denominators at ev
    ! recover physical rho. Core numerator conversions also cancel at ev.
    ! deltaEv is applied by the consumer, never passed into core diagrams.
    if(linear==1) then
      call atomx_val_linear_value_fock(ev,zero_s,d,rs,rd)
    else
      call atomx_val_map_full_fock(ev,zero_s,d,zero_s,zero_d,rs,rd,prime_s,prime_d)
    endif
    ! OUTIN in the above map has already consumed the double numerators.
    ! term31 expects physical singles, not their energy-weighted numerators.
    rvi=s; drvi=0d0; xvo=0d0; rvo=0d0
    call pci_val_term31
    rd=rd+xvo
    rs=rs+matmul(offdiag(:,:,vpartial),s)
    ! [F,S1] also contains -S_mv F_vb a_m^dagger a_b. On the attached
    ! reference this is the double with particle n=v. Its direct rank0
    ! rho(v,m,v,b) is -[j_v]*S_mv*F_vb. The scalar rank0 direct contraction
    ! keeps the reference valence magnetic state in the first particle.
    ! Particle antisymmetrization supplies the other order. No S1 in CI map.
    do b=1,nn
      if(core_partial(b)/=vpartial) cycle
      c=ipxv(b,vpartial,vpartial,0)
      if(c<=0) cycle
      do m=first_virtual(vpartial),nmaxx(vpartial)
        rd(vslot,m,c)=rd(vslot,m,c)-2*abs(vk)*s(m)*fock(vslot,core_slot(b),vpartial)
      enddo
    enddo
    if(linear/=1) rs=rs+offdiag(:,vslot,vpartial)
  end subroutine
end module
