program pci_angular_probe
  implicit none
  integer :: k1,k2,rank,stat
  real(kind=8) :: s
  external s
  do
    read(*,*,iostat=stat) k1,k2,rank
    if (stat/=0) exit
    write(*,'(3I5,ES26.17)') k1,k2,rank,s(rank,k1,k2)
  end do
end program
