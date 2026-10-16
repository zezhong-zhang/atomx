#!/usr/bin/env python3
"""Build a pinned pCI core-SD diagram oracle using native Coulomb features.

The original GPL-3.0 diagram routines are retained verbatim. Only their
radial YFUN/RINT services are replaced by the exact native weighted cell
action. The case-specific direct core library retains independent equation
anchors and an explicit source/dimension ABI; valence SD is not included.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIN = "4d685d82bfdf27876098a87808dadc5fcf3038d1"
SOURCE_HASH = "c39f88b2228ef213eb4a1e860d1bdeadaffe9bceea2537426e312ddf3642c213"
REFERENCE_FILES = {
    "lib/all-order/allcore-ci.f": SOURCE_HASH,
    "lib/all-order/include/d6j.f": "3a709ed9518f816bdfa49daebb2b86585d611c6259e689ab33b4f97fa9f85a97",
    "lib/all-order/include/libD.f": "c37368812edcc3ce04f28fb2bcdd25cd52fe73cdcd1ace0bd00f205474340c14",
    "LICENSE.md": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    source,
    directory,
    dimensions,
    compiler,
    *,
    runtime_directory=None,
    blas_library=None,
    term21_threads=0,
    term21_mode="compact",
    core_singles_blas=False,
    core_single_to_doubles_blas=False,
):
    source, directory = Path(source), Path(directory)
    upstream = source / "lib/all-order/allcore-ci.f"
    if any(
        digest(source / name) != expected for name, expected in REFERENCE_FILES.items()
    ):
        raise ValueError(
            "pCI diagram/angular source or license differs from the pinned reference"
        )
    if directory.exists():
        raise FileExistsError("use a fresh case-specific build directory")
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1
        for v in dimensions.values()
    ):
        raise ValueError("positive integer complete case dimensions required")
    required = {
        "NHF",
        "NX",
        "NS",
        "KX1",
        "NXX",
        "NK",
        "NN",
        "KK",
        "NCH",
        "NXK",
        "NV",
        "NG",
        "NE",
    }
    if dimensions.keys() != required or dimensions["KK"] < dimensions["KX1"]:
        raise ValueError(
            "complete case dimensions and sufficient multipole storage required"
        )
    if (
        isinstance(term21_threads, bool)
        or not isinstance(term21_threads, int)
        or term21_threads < 0
    ):
        raise ValueError("nonnegative integer term21 worker count required")
    term21_threads = min(term21_threads, dimensions["NN"] * (dimensions["NN"] + 1) // 2)
    if term21_mode not in ("compact", "streamed"):
        raise ValueError("term21 mode must be compact or streamed")
    if not isinstance(core_singles_blas, bool) or (
        core_singles_blas and not term21_threads
    ):
        raise ValueError("BLAS core singles requires an explicit positive worker count")
    if core_singles_blas and blas_library is None:
        raise ValueError("BLAS core singles requires an explicit LP64 BLAS library")
    if not isinstance(core_single_to_doubles_blas, bool) or (
        core_single_to_doubles_blas and (not term21_threads or blas_library is None)
    ):
        raise ValueError(
            "BLAS single-to-doubles requires boolean, positive workers and LP64 BLAS"
        )
    if term21_threads and term21_mode == "streamed" and blas_library is None:
        raise ValueError("streamed term21 requires an explicit LP64 BLAS library")
    blas_probe = None
    if blas_library is not None:
        blas_library = Path(blas_library).resolve(strict=True)
        environment = os.environ.copy()
        paths = [str(blas_library.parent)]
        if runtime_directory is not None:
            paths.append(str(Path(runtime_directory).resolve(strict=True)))
        if environment.get("LD_LIBRARY_PATH"):
            paths.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = ":".join(paths)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes,json,sys; b=ctypes.CDLL(sys.argv[1]); "
                "b.openblas_get_config.restype=ctypes.c_char_p; "
                "print(json.dumps(dict(config=b.openblas_get_config().decode(), "
                "parallel=b.openblas_get_parallel(),threads=b.openblas_get_num_threads())))",
                str(blas_library),
            ],
            text=True,
            capture_output=True,
            check=True,
            env=environment,
        )
        blas_probe = json.loads(probe.stdout)
        if "USE64BITINT" in blas_probe["config"]:
            raise ValueError("native SD BLAS requires Fortran LP64, not ILP64")
        if (
            term21_threads > 1
            and (
                term21_mode == "streamed"
                or core_singles_blas
                or core_single_to_doubles_blas
            )
            and blas_probe["parallel"] != 2
        ):
            raise ValueError(
                "parallel SD BLAS requires an OpenMP OpenBLAS build; serial caller concurrency is unsupported"
            )
    directory.mkdir(parents=True)
    shutil.copy2(source / "LICENSE.md", directory / "pCI_LICENSE.md")
    shutil.copy2(upstream, directory / "allcore-ci.original.f")
    text = upstream.read_text()
    body = text[
        text.index("      subroutine rhocore") : text.index('      include "d6j.f"')
    ]
    (directory / "core_diagrams.f").write_text(
        body + '\n      include "d6j.f"\n      include "libD.f"\n'
    )
    for name in ("d6j.f", "libD.f"):
        shutil.copy2(source / "lib/all-order/include" / name, directory / name)
    d = dimensions
    global_values = dict(NHF=d["NHF"], NGP=d["NHF"], NX=d["NX"], NS=d["NS"], KX=d["KK"])
    (directory / "global.par").write_text(
        "".join(f"      PARAMETER ({k}={v})\n" for k, v in global_values.items())
    )
    other = {k: d[k] for k in ("KX1", "NXX", "NK", "NN", "KK", "NCH", "NXK")}
    other.update(
        NL=2 * d["NX"],
        NKX=d["NX"] + d["KK"],
        NREC=8 * 2 * d["NX"] * (d["NHF"] + 1),
        NXV=1,
        NVH=1,
        NNH=d["NN"] ** 2,
        nit=10,
        nyy=1,
    )
    (directory / "all.par").write_text(
        '      include "global.par"\n'
        + "".join(f"      PARAMETER ({k}={v})\n" for k, v in other.items())
    )
    wrapper = """module native_sd_bridge
  use iso_c_binding
  implicit none
  integer NHF,NGP,NX,NS,KX,KX1,NXX,NK,NN,KK,NCH,NXK,NL,NKX,NREC,NXV,NVH,NNH,nit,nyy
  include 'all.par'
  integer, parameter :: nv=CASE_NV, ng=CASE_NG, ne=CASE_NE
  real(c_double) gg(nhf,nx,kx1),ff(nhf,nx,kx1),ee(nx,kx1)
  common /spl/ gg,ff,ee
  real(c_double) r(nhf),rp(nhf),rpor(nhf),h
  integer maxgrid
  common /radial/ r,rp,rpor,h,maxgrid
  real(c_double) go(nhf,ns),fo(nhf,ns),eo(ns)
  integer no(ns),ko(ns),ncore
  common /core/ go,fo,eo,no,ko,ncore
  integer nmax,lmax,nmax1,lmax1,nmaxx(nx),inmax
  common /ldata/ nmax,lmax,nmax1,lmax1
  common /nmaxx/ nmaxx
  common /x12ini/ inmax
  integer ipx1(nn,nn,nk,nk,0:kk),ipx2(nn,nn,nk,nk,0:kk),ichan1,ichan2
  real(c_double) x1(nxx,nxx,nxk),x2(nxx,nxx,nxk)
  common /xcore1/ ipx1,x1,ichan1
  common /xcore2/ ipx2,x2,ichan2
  integer ipxc(nn,nn,nk,nk,0:kk),icchan
  common /irhoc/ ipxc,icchan
  real(c_double) xci(nxx,nxx,nch),rmi(nxx,nn),xco(nxx,nxx,nch),rmo(nxx,nn)
  common /rhocin/ xci,rmi
  common /rhocout/ xco,rmo
  integer ipa(nch),ipb(nch),ipm(nch),ipn(nch),ipl(nch)
  common /ips/ ipa,ipb,ipm,ipn,ipl
  real(c_double) factors(100)
  common /clebsc/ factors
  integer(c_int) outer(2,nv,ng),edge_left(ne),edge_right(ne)
  real(c_double) rising(nv,ng,0:kk),falling(nv,ng,0:kk),edge_weight(ne,0:kk)
  real(c_double) fock(nx,nx,kx1),offdiag(nx,nx,kx1),alpha(nk,nk,0:kk),beta(nk,nk,0:kk)
  integer core_slot(nn),core_partial(nn),first_virtual(nk)
  real(c_double) singles_work(nxx,nn),doubles_work(nxx,nxx,nch)
  logical :: diagram_progress=.false.
  integer(c_int64_t) progress_tick,progress_rate
contains
  subroutine atomx_sd_set_progress(enabled) bind(C)
    integer(c_int),value,intent(in) :: enabled
    diagram_progress=enabled/=0
  end subroutine
  subroutine log_diagram(name)
    use iso_fortran_env, only: error_unit
    character(*),intent(in) :: name
    integer(c_int64_t) after
    if(.not.diagram_progress) return
    call system_clock(after)
    write(error_unit,'(a,a,a,f12.3)') 'native SD stage ',name,' seconds=',real(after-progress_tick,c_double)/progress_rate
    flush(error_unit)
    progress_tick=after
  end subroutine
  subroutine atomx_sd_init(p,q,energies,slots,kappas,limits,radii,om,el,er,rw,fw,ew) bind(C)
    real(c_double), intent(in) :: p(nhf,nx,kx1),q(nhf,nx,kx1),energies(nx,kx1),radii(nhf)
    integer(c_int), intent(in) :: slots(nn),kappas(nn),limits(nx),om(2,nv,ng),el(ne),er(ne)
    real(c_double), intent(in) :: rw(nv,ng,0:kk),fw(nv,ng,0:kk),ew(ne,0:kk)
    integer i,ind,k,kap,l,j,n0
    gg=p; ff=q; ee=energies; r=radii; rp=1d0; rpor=0d0; h=1d0; maxgrid=nhf
    nmax=nxx; lmax=(kx1-1)/2; nmax1=nmax; lmax1=lmax; inmax=kx1; nmaxx=limits
    ncore=nn; no=0; ko=0; go=0d0; fo=0d0; eo=0d0
    ipx1=0; ipx2=0; ipxc=0; x1=0d0; x2=0d0
    xco=0d0; xci=0d0; rmo=0d0; rmi=0d0
    outer=om; edge_left=el; edge_right=er; rising=rw; falling=fw; edge_weight=ew
    factors(1)=1d0
    do i=2,100
      factors(i)=factors(i-1)*(i-1)
    enddo
    do i=1,nn
      k=kappas(i)
      call klj(k,kap,l,j,ind,n0)
      no(i)=slots(i)+n0; ko(i)=k
      core_slot(i)=slots(i); core_partial(i)=ind
      go(:,i)=p(:,slots(i),ind); fo(:,i)=q(:,slots(i),ind); eo(i)=energies(slots(i),ind)
    enddo
    call rhocore
    call rhoini(0)
    do i=1,nk
      call indk1(i,k,kap,l,j,n0)
      call st(k,first_virtual(i))
    enddo
  end subroutine

  subroutine atomx_sd_set_fock(f,a,b) bind(C)
    real(c_double), intent(in) :: f(nx,nx,kx1),a(nk,nk,0:kk),b(nk,nk,0:kk)
    integer i,k
    fock=f; offdiag=f; alpha=a; beta=b
    do k=1,kx1
      do i=1,nx
        offdiag(i,i,k)=0d0
      enddo
    enddo
  end subroutine

  subroutine atomx_sd_info(channels,initial,source,singles,counts) bind(C)
    integer(c_int), intent(out) :: channels(5,nch),counts(3)
    real(c_double), intent(out) :: initial(nxx,nxx,nch),source(nxx,nxx,nch),singles(nxx,nn)
    integer c,ix
    channels(1,:)=ipa; channels(2,:)=ipb; channels(3,:)=ipm; channels(4,:)=ipn; channels(5,:)=ipl
    initial=xco; singles=rmo; source=0d0
    do c=1,icchan
      ix=ipx1(ipa(c),ipb(c),ipm(c),ipn(c),ipl(c))
      if(ix>0) source(:,:,c)=x1(:,:,ix)
    enddo
    counts=(/icchan,ichan1,ichan2/)
  end subroutine

  subroutine atomx_sd_action(ts,td,rs,rd) bind(C)
    real(c_double), intent(in) :: ts(nxx,nn),td(nxx,nxx,nch)
    real(c_double), intent(out) :: rs(nxx,nn),rd(nxx,nxx,nch)
    rmo=ts; xco=td
    call outin
    call coulomb_action()
    rs=rmo; rd=xco
  end subroutine

  subroutine coulomb_action()
    if(diagram_progress) call system_clock(progress_tick,progress_rate)
    call terma1
    call terma2
    call terma3
    call terma4
    call term1
    call term21
    call term31
    call term32
    call term41
    call term42
    call term5(1)
    call term5(2)
  end subroutine

  subroutine profiled_coulomb_action(times)
    real(c_double), intent(out) :: times(12)
    integer(c_int64_t) before,after,rate
    call system_clock(before,rate)
    call terma1
    call system_clock(after); times(1)=real(after-before,c_double)/rate; before=after
    call terma2
    call system_clock(after); times(2)=real(after-before,c_double)/rate; before=after
    call terma3
    call system_clock(after); times(3)=real(after-before,c_double)/rate; before=after
    call terma4
    call system_clock(after); times(4)=real(after-before,c_double)/rate; before=after
    call term1
    call system_clock(after); times(5)=real(after-before,c_double)/rate; before=after
    call term21
    call system_clock(after); times(6)=real(after-before,c_double)/rate; before=after
    call term31
    call system_clock(after); times(7)=real(after-before,c_double)/rate; before=after
    call term32
    call system_clock(after); times(8)=real(after-before,c_double)/rate; before=after
    call term41
    call system_clock(after); times(9)=real(after-before,c_double)/rate; before=after
    call term42
    call system_clock(after); times(10)=real(after-before,c_double)/rate; before=after
    call term5(1)
    call system_clock(after); times(11)=real(after-before,c_double)/rate; before=after
    call term5(2)
    call system_clock(after); times(12)=real(after-before,c_double)/rate
  end subroutine

  subroutine atomx_sd_energy(ts,td,result) bind(C)
    real(c_double), intent(in) :: ts(nxx,nn),td(nxx,nxx,nch)
    real(c_double), intent(out) :: result
    rmo=ts; xco=td
    call energy(result)
  end subroutine

  real(c_double) function rho_value(m,n,i,j,im,jn,k) result(value)
    integer, intent(in) :: m,n,i,j,im,jn,k
    integer c
    value=0d0
    if(i>=j) then
      c=ipxc(i,j,im,jn,k)
      if(c>0) value=xci(m,n,c)
    else
      c=ipxc(j,i,jn,im,k)
      if(c>0) value=xci(n,m,c)
    endif
  end function

  subroutine atomx_sd_fock_action(ts,td,rs,rd) bind(C)
    real(c_double), intent(in) :: ts(nxx,nn),td(nxx,nxx,nch)
    real(c_double), intent(out) :: rs(nxx,nn),rd(nxx,nxx,nch)
    ! This binding returns only the noncanonical one-body contribution.
    rmi=ts; xci=td
    call fock_contribution(rs,rd)
  end subroutine

  subroutine fock_contribution(rs,rd)
    real(c_double), intent(out) :: rs(nxx,nn),rd(nxx,nxx,nch)
    integer c,i,j,im,jn,k,m,n,ic,jc,ij
    rs=0d0; rd=0d0
    do c=1,icchan
      i=ipa(c); j=ipb(c); im=ipm(c); jn=ipn(c); k=ipl(c)
      ic=core_partial(i); jc=core_partial(j)
      ! Same-kappa virtual Fock action, including all off-diagonal entries.
      rd(:,:,c)=matmul(offdiag(:nxx,:nxx,im),xci(:,:,c)) &
                +matmul(xci(:,:,c),transpose(offdiag(:nxx,:nxx,jn)))
      do ij=1,nn
        do n=first_virtual(jn),nmaxx(jn)
          do m=first_virtual(im),nmaxx(im)
            if(core_partial(ij)==ic.and.ij/=i) rd(m,n,c)=rd(m,n,c) &
              -fock(core_slot(ij),core_slot(i),ic)*rho_value(m,n,ij,j,im,jn,k)
            if(core_partial(ij)==jc.and.ij/=j) rd(m,n,c)=rd(m,n,c) &
              -fock(core_slot(ij),core_slot(j),jc)*rho_value(m,n,i,ij,im,jn,k)
          enddo
        enddo
      enddo
      ! Padded core entries are not amplitude coordinates.
      rd(:first_virtual(im)-1,:,c)=0d0
      rd(:,:first_virtual(jn)-1,c)=0d0
      if(nmaxx(im)<nxx) rd(nmaxx(im)+1:,:,c)=0d0
      if(nmaxx(jn)<nxx) rd(:,nmaxx(jn)+1:,c)=0d0
    enddo
    do i=1,nn
      im=core_partial(i)
      rs(:,i)=matmul(offdiag(:nxx,:nxx,im),rmi(:,i))
      do m=first_virtual(im),nmaxx(im)
        rs(m,i)=rs(m,i)+fock(m,core_slot(i),im)
        do j=1,nn
          jn=core_partial(j)
          if(jn==im.and.j/=i) rs(m,i)=rs(m,i)-fock(core_slot(j),core_slot(i),im)*rmi(m,j)
          do n=first_virtual(jn),nmaxx(jn)
            do k=0,kk
              rs(m,i)=rs(m,i)+fock(core_slot(j),n,jn) &
                *(alpha(im,jn,k)*rho_value(m,n,i,j,im,jn,k) &
                  -beta(im,jn,k)*rho_value(n,m,i,j,jn,im,k))
            enddo
          enddo
        enddo
      enddo
      rs(:first_virtual(im)-1,i)=0d0
      if(nmaxx(im)<nxx) rs(nmaxx(im)+1:,i)=0d0
    enddo
  end subroutine

  subroutine vector_unpack(total,t)
    integer(c_int64_t), intent(in) :: total
    real(c_double), intent(in) :: t(total)
    integer i,c,im,jn,m,n
    integer(c_int64_t) ix
    ix=0; rmo=0d0; xco=0d0
    do i=1,nn
      im=core_partial(i)
      do m=first_virtual(im),nmaxx(im)
        ix=ix+1; rmo(m,i)=t(ix)
      enddo
    enddo
    do c=1,icchan
      im=ipm(c); jn=ipn(c)
      do n=first_virtual(jn),nmaxx(jn)
        do m=first_virtual(im),nmaxx(im)
          ix=ix+1; xco(m,n,c)=t(ix)
        enddo
      enddo
    enddo
    if(ix/=total) error stop 'native SD complete coordinate count mismatch'
  end subroutine

  subroutine atomx_sd_vector_info(total,source,denominator) bind(C)
    integer(c_int64_t), value :: total
    real(c_double), intent(out) :: source(total),denominator(total)
    integer i,j,c,im,jn,m,n,ic,jc,k,xidx
    integer(c_int64_t) ix
    ix=0
    do i=1,nn
      im=core_partial(i)
      do m=first_virtual(im),nmaxx(im)
        ix=ix+1
        source(ix)=fock(m,core_slot(i),im)
        denominator(ix)=eo(i)-ee(m,im)
      enddo
    enddo
    do c=1,icchan
      i=ipa(c); j=ipb(c); im=ipm(c); jn=ipn(c); k=ipl(c)
      xidx=ipx1(i,j,im,jn,k)
      do n=first_virtual(jn),nmaxx(jn)
        do m=first_virtual(im),nmaxx(im)
          ix=ix+1; source(ix)=0d0
          if(xidx>0) source(ix)=x1(m,n,xidx)
          denominator(ix)=eo(i)+eo(j)-ee(m,im)-ee(n,jn)
        enddo
      enddo
    enddo
    if(ix/=total) error stop 'native SD complete coordinate count mismatch'
  end subroutine

  subroutine atomx_sd_vector_action(total,t,result) bind(C)
    integer(c_int64_t), value :: total
    real(c_double), intent(in) :: t(total)
    real(c_double), intent(out) :: result(total)
    call vector_unpack(total,t)
    call outin
    call coulomb_action()
    call fock_contribution(singles_work,doubles_work)
    call vector_result(total,result)
  end subroutine

  subroutine atomx_sd_profile_vector_action(total,t,result,times) bind(C)
    integer(c_int64_t), value :: total
    real(c_double), intent(in) :: t(total)
    real(c_double), intent(out) :: result(total),times(16)
    integer(c_int64_t) before,after,rate
    call system_clock(before,rate)
    call vector_unpack(total,t)
    call system_clock(after); times(1)=real(after-before,c_double)/rate; before=after
    call outin
    call system_clock(after); times(2)=real(after-before,c_double)/rate
    call profiled_coulomb_action(times(3:14))
    call system_clock(before)
    call fock_contribution(singles_work,doubles_work)
    call system_clock(after); times(15)=real(after-before,c_double)/rate; before=after
    call vector_result(total,result)
    call system_clock(after); times(16)=real(after-before,c_double)/rate
  end subroutine

  subroutine vector_result(total,result)
    integer(c_int64_t), intent(in) :: total
    real(c_double), intent(out) :: result(total)
    integer i,j,c,im,jn,m,n
    integer(c_int64_t) ix
    real(c_double) denominator
    ix=0
    do i=1,nn
      im=core_partial(i)
      do m=first_virtual(im),nmaxx(im)
        ix=ix+1; denominator=eo(i)-ee(m,im)
        result(ix)=denominator*rmi(m,i)-rmo(m,i)-singles_work(m,i)+fock(m,core_slot(i),im)
      enddo
    enddo
    do c=1,icchan
      i=ipa(c); j=ipb(c); im=ipm(c); jn=ipn(c)
      do n=first_virtual(jn),nmaxx(jn)
        do m=first_virtual(im),nmaxx(im)
          ix=ix+1; denominator=eo(i)+eo(j)-ee(m,im)-ee(n,jn)
          result(ix)=denominator*xci(m,n,c)-xco(m,n,c)-doubles_work(m,n,c)
        enddo
      enddo
    enddo
    if(ix/=total) error stop 'native SD complete coordinate count mismatch'
  end subroutine

  subroutine atomx_sd_vector_energy(total,t,result) bind(C)
    integer(c_int64_t), value :: total
    real(c_double), intent(in) :: t(total)
    real(c_double), intent(out) :: result
    integer i,m,ind
    call vector_unpack(total,t)
    call energy(result)
    do i=1,nn
      ind=core_partial(i)
      do m=first_virtual(ind),nmaxx(ind)
        result=result+2*abs(ko(i))*fock(core_slot(i),m,ind)*rmo(m,i)
      enddo
    enddo
  end subroutine

  subroutine atomx_sd_native_energy(ts,td,result) bind(C)
    real(c_double), intent(in) :: ts(nxx,nn),td(nxx,nxx,nch)
    real(c_double), intent(out) :: result
    integer i,m,ind
    rmo=ts; xco=td
    call energy(result)
    do i=1,nn
      ind=core_partial(i)
      do m=first_virtual(ind),nmaxx(ind)
        result=result+2*abs(ko(i))*fock(core_slot(i),m,ind)*ts(m,i)
      enddo
    enddo
  end subroutine

  recursive subroutine density_action(x,y,k)
    real(c_double), intent(in) :: x(nhf)
    real(c_double), intent(out) :: y(nhf)
    integer, intent(in) :: k
    real(c_double) up(nv),down(nv),prefix,suffix,w
    integer cell,c,m,e,i,j
    y=0d0; up=0d0; down=0d0
    do m=1,ng
      do cell=1,nv
        do c=1,2
          i=outer(c,cell,m)
          up(cell)=up(cell)+rising(cell,m,k)*x(i)
          down(cell)=down(cell)+falling(cell,m,k)*x(i)
        enddo
      enddo
    enddo
    prefix=0d0
    do cell=1,nv
      do m=1,ng
        do c=1,2
          i=outer(c,cell,m)
          y(i)=y(i)+falling(cell,m,k)*prefix
        enddo
      enddo
      prefix=prefix+up(cell)
    enddo
    suffix=0d0
    do cell=nv,1,-1
      do m=1,ng
        do c=1,2
          i=outer(c,cell,m)
          y(i)=y(i)+rising(cell,m,k)*suffix
        enddo
      enddo
      suffix=suffix+down(cell)
    enddo
    do e=1,ne
      i=edge_left(e); j=edge_right(e); w=edge_weight(e,k)
      y(i)=y(i)+w*x(j); y(j)=y(j)+w*x(i)
    enddo
  end subroutine
end module

subroutine yfun(x,y,k,m,*)
  use native_sd_bridge, only: nhf,kk,density_action
  implicit none
  integer k,m
  double precision x(nhf),y(nhf)
  if(k<0.or.k>kk.or.m/=nhf) error stop 'invalid complete native density action'
  call density_action(x,y,k)
end subroutine

double precision function rint(f,na,nb,nq,h)
  implicit none
  integer na,nb,nq
  double precision f(*),h
  rint=sum(f(na:nb))
end function
"""
    for key in ("NV", "NG", "NE"):
        wrapper = wrapper.replace("CASE_" + key, str(d[key]))
    blas_identity = None
    # Logging is outside the diagram routines and does not change their data
    # or equations. It is disabled by default, including in timing controls.
    for stage in (
        "terma1",
        "terma2",
        "terma3",
        "terma4",
        "term1",
        "term21",
        "term31",
        "term32",
        "term41",
        "term42",
        "term5(1)",
        "term5(2)",
    ):
        call = f"    call {stage}\n"
        # Only the ordinary action gets logging. The separate profiler has
        # its original stage timers and identical numerical call sequence.
        begin = wrapper.index("  subroutine coulomb_action()")
        end = wrapper.index("  subroutine profiled_coulomb_action(times)")
        fragment = wrapper[begin:end].replace(
            call, call + f"    call log_diagram('{stage}')\n"
        )
        wrapper = wrapper[:begin] + fragment + wrapper[end:]
    if blas_library is not None:
        blas_library = Path(blas_library).resolve(strict=True)
        template = (
            Path(__file__).resolve().parents[1] / "src/atomx/methods/_pci_core_blas.f90"
        )
        shutil.copy2(template, directory / "core_blas.f90")
        wrapper = wrapper.replace(
            "\n  subroutine atomx_sd_init",
            "\n" + template.read_text() + "\n  subroutine atomx_sd_init",
        )
        wrapper = wrapper.replace("    call rhocore\n", "    call rhocore_blas\n")
        blas_identity = dict(
            path=str(blas_library),
            sha256=digest(blas_library),
            template_sha256=digest(template),
            abi="Fortran LP64 dgemm",
            runtime=blas_probe,
        )
    term21_identity = dict(mode="original pCI term21", threads=0)
    if term21_threads:
        template = (
            Path(__file__).resolve().parents[1]
            / f"src/atomx/methods/_pci_term21_{term21_mode}.f90"
        )
        worker_file = f"term21_{term21_mode}.f90"
        (directory / worker_file).write_text(
            template.read_text().replace("CASE_THREADS", str(term21_threads))
        )
        wrapper = wrapper.replace(
            "    call term21\n", f"    call term21_{term21_mode}\n"
        )
        term21_identity = dict(
            mode=(
                "exact component-tagged core-pair term21"
                if term21_mode == "compact"
                else "exact shared virtual-pair streamed term21"
            ),
            threads=term21_threads,
            template_sha256=digest(template),
            writes="disjoint complete core-pair channels; no reduction",
        )
    (directory / "native_bridge.f90").write_text(wrapper)
    singles_identity = dict(mode="original pCI terma4", workers=0)
    if core_singles_blas:
        template = (
            Path(__file__).resolve().parents[1]
            / "src/atomx/methods/_pci_core_singles_blas.f90"
        )
        (directory / "core_singles_blas.f90").write_text(
            template.read_text().replace("CASE_THREADS", str(term21_threads))
        )
        wrapper = wrapper.replace("    call terma4\n", "    call terma4_blas\n")
        (directory / "native_bridge.f90").write_text(wrapper)
        singles_identity = dict(
            mode="exact component-tagged BLAS terma4",
            workers=term21_threads,
            template_sha256=digest(template),
        )
    single_to_doubles_identity = dict(mode="original pCI term31/32", workers=0)
    if core_single_to_doubles_blas:
        template = (
            Path(__file__).resolve().parents[1]
            / "src/atomx/methods/_pci_core_single_to_doubles_blas.f90"
        )
        (directory / "core_single_to_doubles_blas.f90").write_text(
            template.read_text().replace("CASE_THREADS", str(term21_threads))
        )
        for which in (31, 32):
            old = f"    call term{which}\n"
            if wrapper.count(old) != 2:
                raise AssertionError(
                    "complete action/profiler term31/32 coverage differs"
                )
            wrapper = wrapper.replace(
                old, f"    call core_single_to_doubles_blas({which})\n"
            )
        (directory / "native_bridge.f90").write_text(wrapper)
        single_to_doubles_identity = dict(
            mode="exact native weighted BLAS term31/32",
            workers=term21_threads,
            template_sha256=digest(template),
            writes="disjoint complete core-pair channels; no reduction or amplitude screening",
        )
    flags = [
        str(compiler),
        "-fPIC",
        "-O3",
        "-std=legacy",
        "-ffixed-line-length-none",
        "-ffree-line-length-none",
        "-fno-align-commons",
        "-mcmodel=medium",
        "-static-libgcc",
    ]
    commands = []
    if term21_threads:
        # OpenMP implies recursive locals. Keep large original diagram arrays
        # static; only the explicitly heap-backed worker is compiled with it.
        commands.extend(
            [
                flags + ["-c", filename]
                for filename in ("native_bridge.f90", "core_diagrams.f")
            ]
        )
        commands.append(flags + ["-fopenmp", "-c", worker_file])
        if core_singles_blas:
            commands.append(flags + ["-fopenmp", "-c", "core_singles_blas.f90"])
        if core_single_to_doubles_blas:
            commands.append(
                flags + ["-fopenmp", "-c", "core_single_to_doubles_blas.f90"]
            )
        command = flags + [
            "-shared",
            "-fopenmp",
            "native_bridge.o",
            "core_diagrams.o",
            f"term21_{term21_mode}.o",
            "-o",
            "native_core_sd.so",
        ]
        if core_singles_blas:
            command.append("core_singles_blas.o")
        if core_single_to_doubles_blas:
            command.append("core_single_to_doubles_blas.o")
    else:
        command = flags + [
            "-shared",
            "native_bridge.f90",
            "core_diagrams.f",
            "-o",
            "native_core_sd.so",
        ]
    if runtime_directory is not None:
        runtime_directory = Path(runtime_directory).resolve(strict=True)
        command += [
            "-L" + str(runtime_directory),
            "-Wl,-rpath," + str(runtime_directory),
        ]
    if blas_library is not None:
        command += [str(blas_library), "-Wl,-rpath," + str(blas_library.parent)]
    commands.append(command)
    with (directory / "compiler.log").open("w") as log:
        for invocation in commands:
            completed = subprocess.run(
                invocation, cwd=directory, text=True, capture_output=True
            )
            log.write(completed.stdout + completed.stderr)
            log.flush()
            completed.check_returncode()
    manifest = dict(
        pin=PIN,
        dimensions=d,
        upstream_sha256=SOURCE_HASH,
        upstream_files_sha256=REFERENCE_FILES,
        command=command,
        compile_commands=commands,
        approximation="linearized core SD DC diagrams and full scalar noncanonical Fock terms",
        radial_service="exact weighted native component-tagged cell Coulomb action",
        initialization="original pCI rhocore"
        if blas_identity is None
        else "exact native X1/X2 BLAS contractions",
        blas=blas_identity,
        term21=term21_identity,
        core_singles=singles_identity,
        core_single_to_doubles=single_to_doubles_identity,
        status="equation oracle: not a production precision assertion",
        files={p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--dimensions", required=True, type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--runtime-directory")
    parser.add_argument("--blas-library", type=Path)
    parser.add_argument("--term21-threads", type=int, default=0)
    parser.add_argument("--core-singles-blas", action="store_true")
    parser.add_argument("--core-single-to-doubles-blas", action="store_true")
    parser.add_argument(
        "--term21-mode", choices=("compact", "streamed"), default="compact"
    )
    args = parser.parse_args()
    build(
        args.source,
        args.directory,
        json.loads(args.dimensions.read_text()),
        args.compiler,
        runtime_directory=args.runtime_directory,
        blas_library=args.blas_library,
        term21_threads=args.term21_threads,
        term21_mode=args.term21_mode,
        core_singles_blas=args.core_singles_blas,
        core_single_to_doubles_blas=args.core_single_to_doubles_blas,
    )
