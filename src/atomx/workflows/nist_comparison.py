"""Read-only native campaign comparison against a caller-selected NIST archive."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from atomx.constants import HARTREE_CM1
from atomx.config.subshell import Subshell
from atomx.reference_data.matching import (
    configuration_has_single_level, level_centroids, observed_vacuum_nm,
)

def completed_pair(job, attempts):
    rows=[attempts.get(grid) for grid in job['grids']]
    if not all(row and row['status']=='done' for row in rows):
        return None
    for row in rows:
        physics=row['specification']['physics']
        if (row['specification']['Z']!=job['z'] or row['configuration']!=job['configuration']
                or row['isotope_a']!=job['isotope']
                or physics['nuclear_atomic_weight']!=job['isotope']
                or row['specification']['corrections']['nuclear_mass_me']!=job['mass']):
            raise ValueError('calculation identity differs from queued input')
    if (rows[0]['specification']['physics']!=rows[1]['specification']['physics']
            or rows[0]['specification']['corrections']!=rows[1]['specification']['corrections']):
        raise ValueError('radial refinement changed the physical model')
    if _screening_reference(rows[0])!=_screening_reference(rows[1]):
        raise ValueError('radial refinement changed the self-energy screening reference')
    return rows


def orbital_identity_issue(rows):
    """Flag crossed radial labels for review; do not score them as matched states."""
    for row in rows:
        groups={}
        for n,k,q in row['specification']['state']['subshells']:
            label=Subshell(n,k,q).label
            energy=row['orbital_energies_ha'].get(label)
            if energy is None or not math.isfinite(energy):
                return 'missing_or_nonfinite_orbital_multiplier'
            groups.setdefault(k,[]).append((n,energy))
        for values in groups.values():
            values.sort()
            if any(a[0]<b[0] and a[1]>=b[1] for a,b in zip(values,values[1:])):
                return 'same_kappa_radial_labels_crossed'
    return None


def interval(lower, upper):
    for a,b in zip(lower,upper):
        if (a['specification']['physics']!=b['specification']['physics']
                or a['specification']['corrections']!=b['specification']['corrections']):
            raise ValueError('transition endpoint Hamiltonians differ')
        if _screening_reference(a)!=_screening_reference(b):
            raise ValueError('transition endpoint self-energy screening references differ')
    gaps=[(b['energy_ha']-a['energy_ha'])*HARTREE_CM1 for a,b in zip(lower,upper)]
    return dict(theory_cm1=gaps[1],grid_change_cm1=gaps[1]-gaps[0],
                lower_grids=[r['grid'] for r in lower],upper_grids=[r['grid'] for r in upper])


def _screening_reference(row):
    return {name:value.get('screening_reference','legacy_reference')
            for name,value in row.get('correction_provenance',{}).items()
            if name.startswith('self_energy')}


def assess_negative_configuration_interval(lower, upper):
    """Compare negative theory averages with complete NIST averages, not a line.

    A positive observed transition between overlapping configurations does not
    require their configuration centroids to have that same ordering.
    """
    if lower['status'] != 'evaluated_complete' or upper['status'] != 'evaluated_complete':
        return dict(ordering_assessment='incomplete_nist_manifold')
    gap=upper['centroid_cm1']-lower['centroid_cm1']
    status=('same_negative_nist_centroid_order' if gap<0 else
            'opposite_nist_centroid_order' if gap>0 else 'degenerate_nist_centroids')
    return dict(ordering_assessment=status,nist_configuration_interval_cm1=gap)


def apply_recovered_pairs(jobs, complete, rejected, recovery_states):
    """Fill unavailable states with whole audited pairs, in caller priority order.

    The caller audits each recovery snapshot before supplying its states. This
    boundary independently verifies queued identity and prevents replacing an
    already accepted state or mixing grids from different calculations.
    """
    queued={(j['z'],j['charge'],j['configuration']):j for j in jobs.values()}
    selected={}
    for entry in recovery_states:
        job=entry['job'];key=(job['z'],job['charge'],job['configuration'])
        if key not in queued:
            raise ValueError('recovered configuration is outside the base campaign')
        original=queued[key]
        for name in ('mass','isotope','is_ground','grids'):
            if job[name]!=original[name]:
                raise ValueError(f'recovered input differs from base campaign: {name}')
        if key in complete:
            continue
        pair=completed_pair(original,{r['grid']:r for r in entry['pair']})
        if pair is None or orbital_identity_issue(pair):
            raise ValueError('recovery pair is incomplete or has unverified orbital identity')
        complete[key]=pair
        rejected.pop(key,None)
        selected[key]=entry['source_id']
    return selected


def collect(campaign,archive,*,recovery_states=()):
    with sqlite3.connect(campaign.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        identity=json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        jobs={i:json.loads(s) for i,s in db.execute('SELECT id,input_json FROM jobs')}
        attempts={i:{} for i in jobs}
        for i,g,s in db.execute('SELECT job_id,grid,result_json FROM attempts'):
            attempts[i][g]=json.loads(s)
    complete={}
    rejected={}
    grounds={}
    for i,job in jobs.items():
        key=(job['z'],job['charge'],job['configuration'])
        pair=completed_pair(job,attempts[i])
        if pair:
            issue=orbital_identity_issue(pair)
            if issue:rejected[key]=issue
            else:complete[key]=pair
        if job['is_ground']:
            stage=key[:2]
            if stage in grounds:raise ValueError('ambiguous queued ground configuration')
            grounds[stage]=key
    recovered_sources=apply_recovered_pairs(jobs,complete,rejected,recovery_states)
    source=Path(identity['settings']['source'])
    elements=identity['settings']['elements']
    placeholders=','.join('?' for _ in elements)
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as ledger:
        mappings=ledger.execute(f'''SELECT z,charge,row_index,lower_configuration,
            upper_configuration,classification_source,observed_nm,medium FROM line_jobs
            WHERE z IN ({placeholders}) ORDER BY z,charge,row_index''',elements).fetchall()
    with sqlite3.connect(archive.resolve().as_uri()+'?mode=ro',uri=True) as ref:
        ref.row_factory=sqlite3.Row
        ref.execute('BEGIN')
        lines={(r['z'],r['charge'],r['row_index']):dict(r) for r in ref.execute(
            f'SELECT * FROM lines WHERE z IN ({placeholders}) AND observed_nm>0',elements)}
        if set(lines)!={tuple(row[:3]) for row in mappings}:
            raise ValueError('observed archive rows and line queue have different keys')
        centroids=level_centroids(ref,{(j['z'],j['charge'],j['configuration']) for j in jobs.values()},max(elements))
        ionization_references = ({(r['z'],r['charge']):dict(r) for r in ref.execute(
            f'SELECT * FROM nist_ionization WHERE z IN ({placeholders})', elements)}
            if ref.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='nist_ionization'").fetchone() else {})
    ionizations = ionization_comparisons(grounds, complete, ionization_references)
    if recovery_states:
        for row in ionizations:
            for prefix, charge in (('lower', row['charge']), ('upper', row['charge']+1)):
                key = grounds.get((row['z'],charge))
                row[prefix+'_calculation_source'] = (recovered_sources.get(key,'base_campaign')
                                                      if key in complete else None)
    comparisons=[]
    for z,q,index,lo,hi,origin,observed,medium in mappings:
        reference=lines[z,q,index]
        if observed!=reference['observed_nm'] or medium!=reference['medium']:
            raise ValueError('queued observation differs from archive')
        row=dict(z=z,charge=q,row_index=index,lower_configuration=lo,upper_configuration=hi,
            classification_source=origin,observed_nm=observed,medium=medium,
            observed_unc_nm=reference['observed_unc_nm'],ritz_nm=reference['ritz_nm'],
            lower_term=reference['lower_term'],upper_term=reference['upper_term'],
            lower_j=reference['lower_j'],upper_j=reference['upper_j'],
            line_reference=reference['line_reference'])
        lower,upper=complete.get((z,q,lo)),complete.get((z,q,hi))
        if recovery_states:
            row['lower_calculation_source']=recovered_sources.get((z,q,lo),'base_campaign') if lower else None
            row['upper_calculation_source']=recovered_sources.get((z,q,hi),'base_campaign') if upper else None
        if not lo or not hi:row['status']='missing_classification'
        elif lo==hi:row['status']='unresolved_intra_configuration'
        elif (z,q,lo) in rejected or (z,q,hi) in rejected:
            row['status']='orbital_identity_unverified'
        elif lower is None or upper is None:row['status']='endpoints_not_ready'
        else:
            row.update(interval(lower,upper))
            vacuum=observed_vacuum_nm(observed,medium)
            if vacuum is None:row['status']='unknown_wavelength_medium'
            elif row['theory_cm1']<=0:
                row['status']='inverted_configuration_order'
                if row['theory_cm1']<0:
                    row.update(assess_negative_configuration_interval(
                        centroids[z,q,lo],centroids[z,q,hi]))
            else:
                row.update(observed_cm1=1e7/vacuum,calculated_vacuum_nm=1e7/row['theory_cm1'])
                row['residual_cm1']=row['theory_cm1']-row['observed_cm1']
                row['status']=('matched_unique_levels' if configuration_has_single_level(lo)
                    and configuration_has_single_level(hi) else 'configuration_average_only')
        comparisons.append(row)
    centroid_rows=[]
    for key,pair in complete.items():
        ground_key=grounds.get(key[:2])
        if key==ground_key:continue
        ground=complete.get(ground_key)
        if ground is None:continue
        row=dict(z=key[0],charge=key[1],configuration=key[2],ground_configuration=ground_key[2],
            nist_upper=centroids[key],nist_ground=centroids[ground_key],**interval(ground,pair))
        if recovery_states:
            row['lower_calculation_source']=recovered_sources.get(ground_key,'base_campaign')
            row['upper_calculation_source']=recovered_sources.get(key,'base_campaign')
        if centroids[key]['status']==centroids[ground_key]['status']=='evaluated_complete':
            value=centroids[key]['centroid_cm1']-centroids[ground_key]['centroid_cm1']
            row.update(status='matched_evaluated_centroids',nist_cm1=value,residual_cm1=row['theory_cm1']-value)
        else:row['status']='unscored_manifold_coverage'
        centroid_rows.append(row)
    scored=[r for r in comparisons if r['status']=='matched_unique_levels']
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),campaign_identity=identity,
        scope='standard observed lines; isotope tables separate; averages excluded from strict scores',
        summary=dict(configurations=len(jobs),configurations_with_two_successful_grids=len(complete)+len(rejected),
            configurations_ready_for_comparison=len(complete),
            configurations_from_recovery=len(recovered_sources),
            observed_lines=len(comparisons),line_statuses=dict(Counter(r['status'] for r in comparisons)),
            centroid_statuses=dict(Counter(r['status'] for r in centroid_rows)),
            ionization_statuses=dict(Counter(r['status'] for r in ionizations)),
            negative_interval_assessments=dict(Counter(r['ordering_assessment']
                for r in comparisons if 'ordering_assessment' in r)),
            rejected_orbital_identities=len(rejected),
            unique_level_rms_cm1=math.sqrt(sum(r['residual_cm1']**2 for r in scored)/len(scored)) if scored else None),
        lines=comparisons,centroids=centroid_rows,ionizations=ionizations,
        rejected_states=[dict(z=k[0],charge=k[1],configuration=k[2],reason=v) for k,v in rejected.items()])


def ionization_comparisons(grounds, complete, references):
    """Compare adjacent charge-state ground pairs without treating averages as terms.

    NIST theoretical/semiempirical limits remain diagnostics. Nuclear identity
    and shared operator requests must match; a one-electron-only mass addition
    is allowed to disappear on the many-electron side and is shown in the ledger.
    """
    from atomx.reference_data.matching import expand_nist_config
    if not references:
        return []
    rows = []
    for (z, charge), key in sorted(grounds.items()):
        reference = references.get((z, charge))
        upper_key = grounds.get((z, charge+1))
        bare = charge == z-1
        lower, upper = complete.get(key), complete.get(upper_key)
        row = dict(z=z, charge=charge, lower_configuration=key[2],
            upper_configuration='' if bare else upper_key[2] if upper_key else None,
            nist_reference=reference, status='endpoints_not_ready',
            interpretation='Ionization of configuration averages unless both configurations have a unique level; NIST isotope matching is not established here.')
        if lower is None or (not bare and upper is None):
            rows.append(row); continue
        if not bare:
            mismatch = False
            for a,b in zip(lower,upper):
                if (a['isotope_a'] != b['isotope_a'] or
                        a['specification']['physics'] != b['specification']['physics']):
                    mismatch = True
                for name in ('nuclear_mass_me','include_breit','breit_mode','se_backend',
                             'include_vp','nms_mode','include_sms','include_hydrogenic_mass_remainder'):
                    if a['specification']['corrections'].get(name) != b['specification']['corrections'].get(name):
                        mismatch = True
                if _screening_reference(a) != _screening_reference(b):
                    mismatch = True
            if mismatch:
                row['status']='endpoint_model_mismatch'; rows.append(row); continue
        gaps=[((0. if bare else upper[i]['energy_ha'])-r['energy_ha'])*HARTREE_CM1
              for i,r in enumerate(lower)]
        row.update(theory_cm1=gaps[1], grid_change_cm1=gaps[1]-gaps[0],
            lower_grids=[r['grid'] for r in lower],
            upper_grids=[] if bare else [r['grid'] for r in upper],
            isotope_a=lower[1]['isotope_a'],
            base_ionization_cm1=((0. if bare else upper[1]['base_energy_ha'])-lower[1]['base_energy_ha'])*HARTREE_CM1)
        shifts_a=lower[1]['corrections']; shifts_b={} if bare else upper[1]['corrections']
        row['correction_differences_cm1']={k:(shifts_b.get(k,0.)-shifts_a.get(k,0.))*HARTREE_CM1
                                         for k in sorted(shifts_a.keys()|shifts_b.keys())}
        row['status']='missing_nist_limit'
        if reference and reference.get('ionization_cm1') is not None:
            matching = expand_nist_config(reference.get('ground_configuration') or '',z-charge)==key[2]
            target = references.get((z,charge+1))
            if not bare:
                matching = matching and bool(target) and expand_nist_config(
                    target.get('ground_configuration') or '',z-charge-1)==upper_key[2]
            if not matching:
                row['status']='nist_ground_configuration_unverified'
            else:
                unique=configuration_has_single_level(key[2]) and (
                    bare or configuration_has_single_level(upper_key[2]))
                kind=reference.get('ionization_kind') or 'unclassified'
                row.update(nist_cm1=reference['ionization_cm1'],
                    nist_uncertainty_cm1=reference.get('uncertainty_cm1'),
                    residual_cm1=gaps[1]-reference['ionization_cm1'],
                    comparison_basis='unique_ground_configurations' if unique else 'configuration_average_vs_ground_level',
                    status=('unique_configuration_vs_evaluated_limit' if unique else
                            'configuration_average_vs_evaluated_limit') if kind=='evaluated' else f'nist_{kind}_limit')
        rows.append(row)
    return rows
