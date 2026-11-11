"""Read-only, aggregate recovery history. No recovery/backend/locking imports."""
from datetime import datetime, timedelta, timezone
from uuid import UUID
import hashlib
import json


def _time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _boot(value):
    try:
        return str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        return None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _lease(record):
    identity = _dict(record.get('identity')) or record
    parts = (identity.get('job_id'), identity.get('lease_id'))
    return parts if all(isinstance(v, str) and v for v in parts) else None


def _key(record):
    identity = _dict(record.get('identity')) or record
    old, new = _boot(identity.get('boot_id')), _boot(record.get('observed_boot'))
    if old and new and old != new:
        return ('boot', new)
    if record.get('kind') == 'previous_boot':
        return None  # A label alone cannot verify a VM reboot.
    if record.get('kind') == 'wallpaper' or identity.get('target') == 'wallpaper':
        lease = _lease(record)
        if lease:
            return ('wallpaper',) + lease
        incident = record.get('incident_id') or record.get('id')
        if isinstance(incident, str) and incident:
            return ('wallpaper_id', incident)
    return None


def _resolution_key(record):
    # A reboot is one counted event, but each affected lease must be resolved.
    lease = _lease(record)
    if lease:
        return ('lease',) + lease
    incident = record.get('incident_id') or record.get('id')
    if isinstance(incident, str) and incident:
        return ('incident', incident)
    return None


def summarize_recovery(state, jobs=(), *, at=None, include_incidents=False):
    """First trustworthy observation per distinct incident; inclusive [now-7d, now].

    Invalid/naive/future dates cannot contribute to the count. Attention is
    independent of the date window. Current job statuses are supplied by the
    caller only for durably retained owners, excluding historical failed jobs.
    This includes the supervisor's legacy unresolved-owner uncertainty state.
    """
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError('aware_snapshot_time_required')
    at = at.astimezone(timezone.utc)
    autos = [r for r in _dict(state.get('automatic_recovery')).values() if isinstance(r, dict)]
    audits = [r for r in state.get('recoveries', []) if isinstance(r, dict)] if isinstance(state.get('recoveries', []), list) else []
    boots = _dict(state.get('boot_startups'))
    # A later verified reboot audit can classify an earlier provisional incident.
    boot_leases = {_lease(r): _key(r) for r in autos + audits
                   if _lease(r) and _key(r) and _key(r)[0] == 'boot'}
    seen, aliases = {}, {}
    def stamp(record, *fields):
        values = [_time(record.get(f)) for f in fields]
        valid = [t for t in values if t is not None and t <= at]
        return min(valid) if valid else None
    def add(key, date, record=None):
        if key is not None and date is not None:
            seen[key] = min(seen.get(key, date), date)
            aliases.setdefault(key, set()).add(('incident',) + key)
            resolution = _resolution_key(record or {})
            if resolution:
                aliases[key].add(('resolution',) + resolution)
    def key(record):
        return boot_leases.get(_lease(record)) or _key(record)
    resolved = {_resolution_key(r) for r in audits if stamp(r, 'recovered_at') is not None}
    attention = (state.get('boot_recovery_status') == 'operator_required' or
                 state.get('recovery_worker_status') == 'operator_required')
    for record in audits:
        add(key(record), stamp(record, 'recovered_at'), record)
    pending = {'observing', 'waiting_execution_deadline', 'stopping', 'stopped',
               'starting', 'verified', 'operator_required'}
    for record in autos:
        incident_key = key(record)
        if not (incident_key and incident_key[0] == 'boot' and _key(record) != incident_key):
            # A provisional wallpaper observation predating the reboot cannot
            # supply that reboot's date. Its later boot audit/startup can.
            add(incident_key, stamp(record, 'created_at', 'term_intent', 'start_intent', 'released_at'), record)
        if record.get('phase') in pending and not (
                _resolution_key(record) is not None and _resolution_key(record) in resolved):
            attention = True
    verified_boot = _boot(state.get('verified_boot'))
    verified_time = _time(_dict(state.get('boot_readiness')).get('observed_at'))
    verified_record = _dict(boots.get(verified_boot))
    verified_time = verified_time or stamp(verified_record, 'created_at', 'start_intent')
    for boot, record in boots.items():
        if not isinstance(record, dict):
            continue
        identity = _dict(record.get('identity'))
        current, previous = _boot(identity.get('boot_id')), _boot(identity.get('previous_verified_boot'))
        date = stamp(record, 'created_at', 'start_intent')
        if current and current == _boot(boot) and previous and current != previous:
            add(('boot', current), date, record)
        # Old boot failures do not stay active after a newer verified boot.
        historical = current == verified_boot or (date and verified_time and date <= verified_time)
        if record.get('phase') in pending and not historical:
            attention = True
    if any(_dict(job).get('recovery_status') == 'operator_required' or
           (_dict(job).get('state') in ('reconciling', 'cleanup_pending') and
            bool(_dict(job).get('error_code')))
           for job in jobs):
        attention = True
    summary = {'schema_version': 1,
            'incidents_7d': sum(at - timedelta(days=7) <= date <= at for date in seen.values()),
            'operator_required': bool(attention), 'snapshot_at': at.isoformat()}
    if include_incidents:
        # Opaque identities only, never raw leases, job IDs, boot IDs or payloads.
        # Resolution aliases preserve acknowledgment when later evidence groups
        # an already observed wallpaper incident into a single reboot incident.
        def opaque(key):
            # Current configuration scope is mutable; historical incident
            # identity must survive a later authorized scope transition.
            return hashlib.sha256(json.dumps(['recovery-incident-v1', key],
                separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        incidents = [{'id': opaque(('incident',) + key), 'first_seen': date.isoformat(),
                      'aliases': sorted(opaque(alias) for alias in aliases[key])}
                     for key, date in seen.items() if at - timedelta(days=7) <= date <= at]
        if len(incidents) > 4096 or sum(len(i['aliases']) for i in incidents) > 8192:
            raise RuntimeError('recovery_incident_inventory_too_large')
        summary['incidents'] = sorted(incidents, key=lambda i: i['id'])
    return summary


def read_recovery_summary(store, *, include_incidents=False):
    state = store.read('ownership.json')
    if not isinstance(state, dict):
        raise RuntimeError('recovery_state_unavailable')
    # Read existing records only. No lock creation, writes, boot/GPU probes or
    # RecoveryManager calls. Atomic journal replacement gives a coherent ledger.
    owners = _dict(state.get('owners'))
    ids = {o.get('job_id') for o in owners.values() if isinstance(o, dict)}
    jobs = [store.read(store.job_name(job_id), {}) for job_id in ids if isinstance(job_id, str)]
    return summarize_recovery(state, jobs, include_incidents=include_incidents)
