"""Archive frozen Runtime V semantics alongside (not in place of) the full goal."""
import json
from pathlib import Path


def session_for(scene):
    binding=getattr(scene,'_v_collection_binding',None)
    if binding is None:return None
    session=getattr(scene.evaluator,'_v_intervention_session',None)
    if session is None or session.config.get('package_id')!=binding['package_id'] or session.branch!=binding['branch']:
        raise RuntimeError('frozen V intervention session is not bound to this collection configuration')
    return session


def begin(scene,episode_dir):
    session=session_for(scene)
    if session is None:return
    session.reset_episode(episode_dir)
    private=Path(episode_dir)/'private';private.mkdir(parents=True,exist_ok=True)
    (private/'v_collection_identity.json').write_text(json.dumps(scene._v_collection_binding,indent=2)+'\n')


def observed(scene,result):
    session=session_for(scene)
    if session is not None and result.get('observation') is not None:
        # Preserve the exact live bindings privately for human-interface replay
        # audits. The public observation still contains only numbered markers.
        runtime=getattr(scene,'_runtime',None)
        snapshot=getattr(getattr(runtime,'store',None),'active_snapshot',None)
        if snapshot is not None:
            bindings=[{'marker':marker,'object_name':binding.object_name,
                       'binding_kind':getattr(binding,'binding_kind',None)}
                      for marker,binding in snapshot.bindings.items()]
            record={'sequence':result.get('sequence'),
                    'observation_id':result['observation'].get('observation_id'),
                    'operation':snapshot.operation,'bindings':bindings}
            with (scene._episode_dir/'private/v_marker_bindings.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        session.observation_delivered()


def finish(scene,reason,goal_snapshot,summary):
    session=session_for(scene)
    if session is None:return
    # Invoke the implementation belonging to this exact frozen Runtime.
    try:
        report=session.finish(reason,goal_snapshot)
    except Exception as exc:
        report={'operational':False,'v_protocol_complete':False,'termination_reason':reason,
                'error_type':type(exc).__name__,'error':str(exc)}
    summary['collection_config_key']=scene._v_collection_binding['collection_config_key']
    summary['v_protocol']=report
    (scene._episode_dir/'private/v_protocol_result.json').write_text(json.dumps(report,indent=2)+'\n')


def require_no_checkpoint(scene):
    if getattr(scene,'_v_collection_binding',None) is not None:
        raise RuntimeError('V checkpoint restore is unavailable until intervention state has a verified serialization contract')


def abort(scene,reason):
    if getattr(scene,'_v_collection_binding',None) is None:return
    private=Path(scene._episode_dir)/'private' if scene._episode_dir is not None else None
    report={'aborted':True,'v_protocol_complete':False,'reason':reason,'same_scene_recovery_allowed':False}
    try:
        session=session_for(scene)
        if scene._last_verifier is not None:
            report['native_report']=session.finish(reason,scene._last_verifier)
        else:
            session.retire()
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
    if private is not None:
        private.mkdir(parents=True,exist_ok=True)
        (private/'v_protocol_abort.json').write_text(json.dumps(report,indent=2)+'\n')
