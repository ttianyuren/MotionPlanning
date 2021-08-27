#!/usr/bin/env python
from run_branch_p1 import *

#######################################################


if __name__ == '__main__':

    with open('exe_plan.pk', 'rb') as f:
        _, exe_plan = pk.load(f)

    connect(use_gui=True)

    PlanningScenario = get_scn(0)
    scn = PlanningScenario(enable_drops=1)

    commands = postprocess_plan(scn, exe_plan)

    saved_world = WorldSaver()
    for i in range(100):
        saved_world.restore()
        time.sleep(3)
        play_commands(commands, use_control=1)
        time.sleep(1)

    disconnect()
