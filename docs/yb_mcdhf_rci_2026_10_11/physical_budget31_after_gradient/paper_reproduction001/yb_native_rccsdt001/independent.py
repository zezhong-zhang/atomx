"""Occupation-list spin algebra and matrix-exponential checks for the pilot."""
import math

import numpy as np
from scipy.linalg import expm


def act(mask, operations):
    occupied=[p for p in range(mask.bit_length()) if mask & (1 << p)]
    phase=1
    for p,create in reversed(operations):
        if (p in occupied)==create:
            return None,0
        phase*=(-1)**sum(q<p for q in occupied)
        if create:
            occupied.append(p)
        else:
            occupied.remove(p)
    return sum(1<<p for p in occupied),phase


def j_squared(labels,masks):
    index={mask:i for i,mask in enumerate(masks)}
    lookup={tuple(label):i for i,label in enumerate(labels)}
    plus=np.zeros((len(masks),len(masks)))
    z=np.array([sum(label[2]/2 for i,label in enumerate(labels) if mask&(1<<i)) for mask in masks])
    for i,(n,k,m) in enumerate(labels):
        if m==2*abs(k)-1:
            continue
        target=lookup[n,k,m+2]
        j=abs(k)-.5
        factor=math.sqrt((j-m/2)*(j+m/2+1))
        for col,mask in enumerate(masks):
            end,sign=act(mask,((target,True),(i,False)))
            if sign:
                plus[index[end],col]+=factor*sign
    return plus.T@plus+np.diag(z*(z+1))


def cluster_matrix(core,masks,reference):
    index={mask:i for i,mask in enumerate(masks)}
    matrix=np.zeros((len(masks),len(masks)))
    triples=np.zeros_like(matrix)
    for amplitude,(holes,particles) in zip(core.amplitudes,core.provenance['excitations'],strict=True):
        ops=tuple((p,True) for p in particles)+tuple((h,False) for h in reversed(holes))
        _,reference_phase=act(reference,ops)
        assert reference_phase in (-1,1)
        for col,mask in enumerate(masks):
            end,phase=act(mask,ops)
            if phase:
                value=amplitude*phase/reference_phase
                matrix[index[end],col]+=value
                if len(holes)==3:
                    triples[index[end],col]+=value
    return matrix,triples


def similarity_check(result,full_h):
    t,t3=cluster_matrix(result.core,result.determinant_masks,result.provenance['reference_mask'])
    ep,em=expm(t),expm(-t)
    hbar=em@full_h@ep-result.core.total_energy_Ha*np.eye(len(full_h))
    lookup={mask:i for i,mask in enumerate(result.determinant_masks)}
    rows=[lookup[mask] for mask in result.projected_masks]
    projected=hbar[np.ix_(rows,rows)]
    return dict(maximum_directed_Hbar_difference_Ha=float(np.max(abs(projected-result.projected_operator_Ha))),
                attached_T3_action_norm=float(np.linalg.norm(t3)),
                non_Hermitian_asymmetry_Ha=float(np.max(abs(result.projected_operator_Ha-result.projected_operator_Ha.T))))


def levels(energies,coefficients,masks,labels,j2):
    occupied=4
    reference=(1<<occupied)-1
    index={mask:i for i,mask in enumerate(masks)}
    output={}
    for key,name,j,multiplicity in [((6,-1),'S',.5,2),((5,2),'D3',1.5,4)]:
        model=[index[reference|(1<<i)] for i,label in enumerate(labels) if label[:2]==key]
        weights=np.sum(abs(coefficients[model])**2,axis=0)
        roots=np.flatnonzero(weights>=.5)
        if len(roots)!=multiplicity:
            raise AssertionError(f'{name}: required complete {multiplicity}-state multiplet, got{len(roots)}')
        vectors=coefficients[:,roots]
        output[name]=dict(roots=roots.tolist(),mean_energy_Ha=float(np.mean(energies[roots])),
            spread_Ha=float(np.ptp(energies[roots])),minimum_bare_model_weight=float(weights[roots].min()),
            J_squared_defect=float(np.linalg.norm(j2@vectors-j*(j+1)*vectors,axis=0).max()))
    return output
