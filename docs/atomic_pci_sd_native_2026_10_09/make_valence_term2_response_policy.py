"""Private experiment: one exact kernel with explicit response computation policy."""

import hashlib
import json
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
source = ROOT / "src/atomx/methods/_pci_valence_term2_streamed.f90"
body = source.read_text()


def change(old, new, count=1):
    global body
    if body.count(old) != count:
        raise AssertionError(f"expected {count} exact response-policy sites: {old}")
    body = body.replace(old, new)


change(
    "subroutine valence_term2_streamed(energy)",
    "subroutine valence_term2_streamed(energy,calculate_response)",
)
change(
    "real(c_double),intent(in) :: energy\n",
    "real(c_double),intent(in) :: energy\n  logical,intent(in) :: calculate_response\n",
)
change(
    "valence_term2_streamed_core(energy,j,im,ir,l,phi,uu,rho,drho, &",
    "valence_term2_streamed_core(energy,calculate_response,j,im,ir,l,phi,uu,rho,drho, &",
    count=2,
)
change(
    "integer,intent(in) :: j,im,ir,l\n",
    "integer,intent(in) :: j,im,ir,l\n  logical,intent(in) :: calculate_response\n",
)
change(
    "          drho(nr,nss)=dxvi(nr,nss,index1)*inverse-amplitude*inverse",
    "          if(calculate_response) drho(nr,nss)=dxvi(nr,nss,index1)*inverse-amplitude*inverse",
)
for call in (
    "      call dgemm('N','T',nhf,cr,cs,1d0,phi(1,nns,iss),nhf, &\n"
    "        drho(nnr,nns),nxx,0d0,dv1(1,nnr),nhf)",
    "        call dgemm('T','N',cm,cn,nhf,1d0,duv(1,nnm),nhf, &\n"
    "          phi(1,nnn,inn),nhf,0d0,dprojected,nxx)",
):
    indent = call[: len(call) - len(call.lstrip())]
    change(
        call, indent + "if(calculate_response) then\n" + call + "\n" + indent + "endif"
    )
change(
    "uv(:,m)=0d0; duv(:,m)=0d0",
    "uv(:,m)=0d0\n        if(calculate_response) duv(:,m)=0d0",
)
for assignment in (
    "          duv(:,m)=duv(:,m)+dv1(:,nr)*uu(:,nr,m)",
    "              dxvo(m+nnm-1,n+nnn-1,index)=dxvo(m+nnm-1,n+nnn-1,index)+dprojected(m,n)*cfin",
):
    indent = assignment[: len(assignment) - len(assignment.lstrip())]
    change(assignment, indent + "if(calculate_response) " + assignment.lstrip())

body += """
subroutine atomx_val_term2_streamed_map(energy,d,dd,rd,pd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
  xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
  call valence_term2_streamed(energy,.true.)
  rd=xvo; pd=dxvo
end subroutine

subroutine atomx_val_term2_value_map(energy,d,rd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh)
  xvi=d; xvo=0d0
  call valence_term2_streamed(energy,.false.)
  rd=xvo
end subroutine

subroutine atomx_val_term2_original_map(energy,d,dd,rd,pd) bind(C)
  use native_valence_bridge
  implicit none
  real(c_double),value,intent(in) :: energy
  real(c_double),intent(in) :: d(nxx,nxx,nvh),dd(nxx,nxx,nvh)
  real(c_double),intent(out) :: rd(nxx,nxx,nvh),pd(nxx,nxx,nvh)
  xvi=d; dxvi=dd; xvo=0d0; dxvo=0d0
  call pci_val_term2(energy)
  rd=xvo; pd=dxvo
end subroutine
"""
target = OWNER / "valence_term2_response_policy.f90"
if target.exists():
    raise FileExistsError("retain prior candidate; do not regenerate over evidence")
target.write_text(body)
(OWNER / "valence_term2_response_policy_contract.json").write_text(
    json.dumps(
        dict(
            hypothesis="Explicit policy guards unconsumed derivative arithmetic in one exact kernel, preserving all signed values and the full requested derivative",
            fixed="all value contractions/angular order/shared complete potentials/full scratch bound unchanged; same finite/full cases as private value candidate",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            signed_gate_Ha=1e-11,
            bitwise_to_retained_candidate=True,
            derivative_gate_Ha=1e-11,
            stop="any signed, response, source or resource mismatch rejects public adoption; no fake derivative returned for value-only mode",
            scope="private prototype; public full kernel unchanged until controls pass",
        ),
        indent=2,
    )
    + "\n"
)
