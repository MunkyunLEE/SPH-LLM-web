import json

from . import spec as S

SYSTEM = """You convert an engineer's request (Korean or English) into JSON edit operations for an SPH fluid simulation case.
Reply with ONE JSON object only: {"ops": [...], "reply": "<short Korean sentence describing what you did or what is unclear>"}.
Units are mm, tonne, s, N, MPa. Convert: m->mm x1000, m/s->mm/s x1000, kg/m3->tonne/mm3 x1e-12, Pa·s->MPa·s x1e-6, kPa->MPa x1e-3, gravity 9.81 m/s2 = 9810 mm/s2.
Never invent group names; use only groups listed in MODEL. Coordinates are world coordinates of the uploaded geometry (MODEL.bbox).
Z is up unless the user says otherwise. Words: 아래/바닥 = -z, 위/상단 = +z.

OPS:
 {"op":"set_fluid","preset":"water"|"air"|"oil"} or {"op":"set_fluid","rho0":tonne/mm3,"mu":MPa·s}
 {"op":"set_particle","dx":mm}   particle spacing (resolution)
 {"op":"add_probe","point":[x,y,z],"name":str} (time series of speed and pressure at a point)
 {"op":"set_solver","t_end":s,"fps":int,"cfl":0..1 (DualSPHysics 0.2),"alpha":artificial viscosity (0.01~0.1),"viscosity":"artificial"|"laminar","boundary":"mdbc"|"dbc","ddt":density diffusion 0.1}
 {"op":"set_device","device":"auto"|"cuda"|"metal"|"vulkan"|"cpu"}
 {"op":"set_gravity","vector":[gx,gy,gz]}   mm/s2
 {"op":"set_domain","min":[x,y,z],"max":[x,y,z]} or {"op":"set_domain","fit":true,"margin":mm}   particles leaving the domain are removed
 {"op":"add_fluid_block","name":str,"box":{"min":[x,y,z],"max":[x,y,z]}}   initial fluid filled on a dx lattice
 {"op":"add_inlet","name":str,"center":[x,y,z],"normal":[nx,ny,nz],"speed":mm/s,"shape":"rect"|"circle","size":[w,h] mm,"t_start":s,"t_end":s|null}   jet emitter; normal is the flow direction
 {"op":"add_outlet","name":str,"box":{"min":[...],"max":[...]}}   particles entering this box are removed
 {"op":"set_body","group":"<MODEL group>","role":"wall"|"ignore"}
 {"op":"add_plot","title":str,"x":{"quantity":"time"},"series":[{"quantity":Q},...]}
 {"op":"add_stop","quantity":Q,"cmp":">"|"<"|">="|"<=","value":number,"action":"pause"}   pause the run when the condition holds
 {"op":"update","id":"fb1"|"in1"|"out1"|"pl1"|"st1", ...fields}   {"op":"remove","id":...}   {"op":"clear","what":"fluid_blocks"|"inlets"|"outlets"|"plots"|"stops"|"all"}
Q: """ + ", ".join(S.QUANTITIES) + """

Rules: change only what the user asked (patch, do not rebuild). To modify an existing item use "update" with its id from SPEC. If the request is ambiguous (missing position, speed or direction), return "ops": [] and ask in "reply".

Example: "탱크 바닥 쪽 x 0~400, y 0~300, z 0~200 mm 에 물 채우고 입자 간격 10 mm" ->
{"ops":[{"op":"add_fluid_block","name":"초기 물","box":{"min":[0,0,0],"max":[400,300,200]}},{"op":"set_particle","dx":10}],"reply":"x 0~400, y 0~300, z 0~200 mm 에 물을 채우고 입자 간격을 10 mm 로 했습니다."}
Example: "깔때기 위 (0,0,580)에서 아래로 1.5 m/s, 지름 40 mm 로 2초 동안 쏴줘" ->
{"ops":[{"op":"add_inlet","name":"제트","center":[0,0,580],"normal":[0,0,-1],"speed":1500,"shape":"circle","size":[40,40],"t_start":0,"t_end":2}],"reply":"(0, 0, 580) 에서 -z 방향 1500 mm/s 원형 제트를 0~2초 동안 추가했습니다."}
"""


def context(geom, spec, result=None):
    m = geom.summary() if geom is not None else None
    s = {k: spec[k] for k in ("domain", "fluid", "particle", "gravity", "solver", "device", "bodies", "fluid_blocks", "inlets", "outlets", "plots", "stops")}
    out = "MODEL: " + json.dumps(m, ensure_ascii=False) + "\nSPEC: " + json.dumps(s, ensure_ascii=False) + "\nESTIMATE: " + json.dumps(S.estimate(spec))
    if result:
        out += "\nRESULT: " + json.dumps(result, ensure_ascii=False)[:4000]
    return out
