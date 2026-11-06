"""Shallow per-candidate evidence, isolated from action success and public feedback."""
from collections.abc import Mapping

def collision_records(payload):
    records=[]
    def visit(value, path, depth):
        if depth>16 or len(records)>=128:return
        if isinstance(value,Mapping):
            collision=value.get('final_state_collision_admission_private')
            if isinstance(collision,Mapping):
                records.append({'sequence':payload.get('sequence'),'source_path':path,
                    'candidate':{k:value[k] for k in ['candidate_id_private','position_m_private','orientation_xyzw_private','initial_aabb_private','official_inside_private','accepted'] if k in value},
                    'collision':dict(collision)})
            for k,v in value.items():
                if k!='final_state_collision_admission_private':visit(v,path+'.'+str(k),depth+1)
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value[:128]):visit(v,path+f'[{i}]',depth+1)
    visit(payload.get('driver_private_diagnostic',{}),'driver_private_diagnostic',0)
    return records
