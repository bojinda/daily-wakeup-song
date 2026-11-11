"""Local operator actions; no remote recovery endpoint, deployment or job trigger."""
import argparse
import json
import os
from .api import Application, serve
from .config import admission_from_config, make_runner
from .core import strict_json, timestamp

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("inspect", "bootstrap", "recover", "serve",
                        "transition-scope", "dual-restore", "migrate-physical-locks",
                        "transition-wallpaper-workflow", "transition-wallpaper-conditioning",
                        "transition-wallpaper-on-demand",
                        "authorize-recovery", "recover-reboot-contract"))
    parser.add_argument("--evidence-file")
    parser.add_argument("--job-id")
    parser.add_argument("--new-config")
    parser.add_argument("--rollback", action='store_true')
    parser.add_argument('--policy-file')
    parser.add_argument('--approval-ref')
    parser.add_argument('--container-review-file')
    args = parser.parse_args()
    if args.action == 'migrate-physical-locks':
        if not args.new_config or not args.evidence_file or not args.job_id or args.rollback:
            parser.error('old/new configs, exact job and migration evidence required; no rollback')
        from .lock_migration import migrate_physical_locks
        with open(args.evidence_file, 'rb') as stream:
            evidence = strict_json(stream.read())
        print(json.dumps(migrate_physical_locks(args.config, args.new_config, args.job_id, evidence)))
        return
    if args.action in ('transition-scope', 'transition-wallpaper-workflow', 'transition-wallpaper-conditioning',
                       'transition-wallpaper-on-demand'):
        if not args.new_config or not args.evidence_file:
            parser.error('new config and local operator evidence required')
        from .scope_transition import (transition_scope, transition_wallpaper_workflow,
                                       transition_wallpaper_conditioning, transition_wallpaper_on_demand)
        with open(args.evidence_file, 'rb') as stream:
            evidence = strict_json(stream.read())
        transition = {'transition-scope': transition_scope,
                      'transition-wallpaper-workflow': transition_wallpaper_workflow,
                      'transition-wallpaper-conditioning': transition_wallpaper_conditioning,
                      'transition-wallpaper-on-demand': transition_wallpaper_on_demand}[args.action]
        print(json.dumps(transition(args.config, args.new_config, evidence, rollback=args.rollback)))
        return
    if args.action == "serve":
        value, runner = make_runner(args.config)
        try:
            token = os.environ.get(value.get("auth_token_env", "AIHUB_GPU_RUNNER_TOKEN"), "")
            if len(token) < 32:
                raise ValueError("high_entropy_internal_token_required")
            listen = value.get("listen", {})
            application = Application(runner, token)
            serve(application, listen.get("host", "127.0.0.1"), listen.get("port", 8790))
        finally:
            runner.close()
        return
    _, targets, admission = admission_from_config(args.config)
    if args.action == 'recover-reboot-contract':
        if (not args.evidence_file or not args.container_review_file or not args.job_id or
                args.new_config or args.rollback or args.policy_file or args.approval_ref):
            parser.error('exact job, operator evidence and private container review required; no policy override')
        from .config import configured_backends
        from .reboot_contract import recover_reboot_contract
        with open(args.evidence_file, 'rb') as stream:
            evidence = strict_json(stream.read())
        with open(args.container_review_file, 'rb') as stream:
            reviewed = strict_json(stream.read())
        if isinstance(reviewed, list) and len(reviewed) == 1:
            reviewed = reviewed[0]
        if not isinstance(evidence, dict) or evidence.get('job_id') != args.job_id:
            parser.error('job ID must match the exact approval')
        print(json.dumps(recover_reboot_contract(admission, targets, configured_backends(targets),
                                                evidence, reviewed)))
        return
    if args.action == 'authorize-recovery':
        if not args.policy_file or not args.approval_ref:
            parser.error('reviewed recovery policy file and deployment approval reference required')
        from .config import configured_backends
        from .recovery import authorize_policy
        with open(args.policy_file, 'rb') as stream:
            policy = strict_json(stream.read())
        result = authorize_policy(admission, targets, configured_backends(targets), policy, args.approval_ref)
        print(json.dumps({'recorded': args.action, 'policy_id': result['policy_id']}))
        return
    if args.action == "inspect":
        print(json.dumps(admission.snapshot(), indent=2))
        return
    if not args.evidence_file:
        parser.error("local operator evidence file is required")
    with open(args.evidence_file, "rb") as stream:
        evidence = strict_json(stream.read())
    if args.action == "bootstrap":
        admission.bootstrap(evidence)
    elif args.action == 'dual-restore':
        if not args.job_id:
            parser.error('exact held job ID required')
        from .dual_gpu import restore_held
        job = admission.store.job(args.job_id)
        proof = restore_held(admission, targets[job['target']], args.job_id, evidence)
        print(json.dumps({'recorded': 'dual-restore', 'owners_retained': True, 'proof': proof}))
        return
    else:
        if not args.job_id:
            parser.error("job ID is required")
        admission.recover(args.job_id, evidence)
        admission.finalize_recoveries(args.job_id)
    print(json.dumps({"recorded": args.action, "backend_action_performed": False}))

if __name__ == "__main__":
    main()
